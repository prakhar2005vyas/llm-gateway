import { useCallback, useEffect, useState } from 'react'

/** Mirrors the JSON from GET /internal/traces (backend/app/routes/internal.py). */
interface Trace {
  id: string
  created_at: string | null
  model: string | null
  status_code: number | null
  latency_ms: number | null
  ttft_ms: number | null
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
  /** Exact Numeric(12,8) serialized as a string; null = cost unknown. */
  cost_usd: string | null
  cache_hit: boolean
  coalesced: boolean
  outcome: string
  error_message: string | null
  test_name: string | null
}

const POLL_MS = 5000

function formatTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return isNaN(d.getTime()) ? '—' : d.toLocaleTimeString([], { hour12: false })
}

function formatCost(cost: string | null): string {
  if (cost === null) return '—'
  const n = Number(cost)
  if (isNaN(n)) return '—'
  return n === 0 ? '$0' : `$${n.toFixed(6)}`
}

function statusColor(status: number | null): string {
  if (status === null) return 'text-gray-500'
  return status < 400 ? 'text-green-400' : 'text-red-400'
}

export default function App() {
  const [traces, setTraces] = useState<Trace[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTraces = useCallback(async () => {
    try {
      // Authorization is injected by the Vite dev-server proxy (vite.config.ts)
      // so the bearer token never enters the browser bundle.
      const res = await fetch('/internal/traces', {
        headers: { 'Content-Type': 'application/json' },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setTraces((await res.json()) as Trace[])
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'fetch failed')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchTraces()
    const id = setInterval(fetchTraces, POLL_MS)
    return () => clearInterval(id)
  }, [fetchTraces])

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">
      <header className="flex items-center justify-between border-b border-gray-800 px-6 py-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">LLM Gateway Traces</h1>
          <p className="text-sm text-gray-400">
            {traces.length} most recent · auto-refreshes every {POLL_MS / 1000}s
          </p>
        </div>
        <button
          onClick={fetchTraces}
          className="rounded-md border border-gray-700 bg-gray-800 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-gray-700 active:bg-gray-600"
        >
          Refresh
        </button>
      </header>

      {error && (
        <div className="mx-6 mt-4 rounded-md border border-red-800 bg-red-950 px-4 py-2 text-sm text-red-300">
          Trace store unreachable ({error}) — the gateway keeps proxying; this
          view recovers automatically.
        </div>
      )}

      <main className="px-6 py-4">
        <div className="overflow-x-auto rounded-lg border border-gray-800">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-800/60 text-xs uppercase tracking-wider text-gray-400">
              <tr>
                <th className="px-4 py-3">Time</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Latency (TTFT)</th>
                <th className="px-4 py-3">Tokens</th>
                <th className="px-4 py-3">Cost</th>
                <th className="px-4 py-3">Flags</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {loading && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                    Loading traces…
                  </td>
                </tr>
              )}
              {!loading && traces.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                    No traces yet — send a request through the gateway.
                  </td>
                </tr>
              )}
              {traces.map((t) => (
                <tr
                  key={t.id}
                  title={t.error_message ?? undefined}
                  className="hover:bg-gray-800/40"
                >
                  <td className="px-4 py-2 font-mono text-gray-300">
                    {formatTime(t.created_at)}
                  </td>
                  <td className="px-4 py-2 text-gray-200">{t.model ?? '—'}</td>
                  <td className={`px-4 py-2 font-mono ${statusColor(t.status_code)}`}>
                    {t.status_code ?? '—'}
                    {t.outcome !== 'ok' && (
                      <span className="ml-2 text-xs text-yellow-500">{t.outcome}</span>
                    )}
                  </td>
                  <td className="px-4 py-2 font-mono text-gray-300">
                    {t.latency_ms !== null ? `${t.latency_ms} ms` : '—'}
                    {t.ttft_ms !== null && (
                      <span className="text-gray-500"> ({t.ttft_ms} ms)</span>
                    )}
                  </td>
                  <td className="px-4 py-2 font-mono text-gray-300">
                    {t.total_tokens ?? '—'}
                  </td>
                  <td className="px-4 py-2 font-mono text-gray-300">
                    {formatCost(t.cost_usd)}
                  </td>
                  <td className="px-4 py-2">
                    {t.cache_hit && (
                      <span className="mr-1 rounded-full border border-green-700 bg-green-900/60 px-2 py-0.5 text-xs font-medium text-green-300">
                        Cache Hit
                      </span>
                    )}
                    {t.coalesced && (
                      <span className="mr-1 rounded-full border border-blue-700 bg-blue-900/60 px-2 py-0.5 text-xs font-medium text-blue-300">
                        Coalesced
                      </span>
                    )}
                    {t.test_name && t.test_name !== 'default' && (
                      <span className="rounded-full border border-purple-700 bg-purple-900/60 px-2 py-0.5 text-xs font-medium text-purple-300">
                        {t.test_name}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  )
}