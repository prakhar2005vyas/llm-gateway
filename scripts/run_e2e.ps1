$ErrorActionPreference = "Stop"

# Navigate to the root directory
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location "$RootDir\.."

Write-Host "--- starting mock upstream provider ---" -ForegroundColor Cyan
$MockProcess = Start-Process -FilePath "python" -ArgumentList "scripts\demo_stub_upstream.py" -PassThru -NoNewWindow

try {
    Write-Host "--- docker compose up --build (detached, using demo config) ---" -ForegroundColor Cyan
    docker compose -f docker-compose.yml -f compose.demo.yml up --build -d

    $HealthUrl = "http://localhost:8000/health"
    $MaxWait = 180 # Increased to 3 minutes because all-MiniLM-L6-v2 is taking over 75s to load into memory
    $Deadline = (Get-Date).AddSeconds($MaxWait)

    Write-Host "--- waiting for gateway to be healthy (up to ${MaxWait}s) ---" -ForegroundColor Cyan
    $IsUp = $false
    while ((Get-Date) -lt $Deadline) {
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $IsUp = $true
                break
            }
        } catch {
            # Ignore errors while waiting
        }
        Start-Sleep -Seconds 1
    }

    if (-not $IsUp) {
        Write-Host "FAIL: gateway did not come up within ${MaxWait}s" -ForegroundColor Red
        docker compose logs gateway
        exit 1
    }

    Write-Host "--- gateway is up, running pytest E2E tests ---" -ForegroundColor Cyan
    
    # Set the test API key
    $env:GATEWAY_API_KEY = "my_secure_local_password"
    
    # Locate and run pytest inside the .venv
    if (Test-Path ".\.venv\Scripts\pytest.exe") {
        & ".\.venv\Scripts\pytest.exe" "scripts\test_e2e.py" "-v"
    } elseif (Test-Path "backend\.venv\Scripts\pytest.exe") {
        & "backend\.venv\Scripts\pytest.exe" "scripts\test_e2e.py" "-v"
    } else {
        pytest "scripts\test_e2e.py" "-v"
    }

    Write-Host "--- E2E tests finished ---" -ForegroundColor Green

} finally {
    Write-Host "--- tearing down ---" -ForegroundColor Cyan
    if ($MockProcess -and -not $MockProcess.HasExited) {
        Stop-Process -Id $MockProcess.Id -Force
    }
    docker compose -f docker-compose.yml -f compose.demo.yml down --volumes --remove-orphans | Out-Null
    Pop-Location
}
