# Changelog

## 2026-08-19: openai/gpt-oss-120b pricing correction
Migrated default model from deprecated llama-3.3-70b-versatile to openai/gpt-oss-120b. BASELINE_PRICES was corrected in seed.py, but the live database's existing model_prices row was NOT auto-updated (seeding is insert-only). Anyone who deployed before this date needs to manually run the UPDATE shown in seed.py's comment, or verify current pricing with:
  docker compose exec postgres psql -U gateway -d gateway -c \
    "SELECT model_id, usd_per_1k_input, usd_per_1k_output FROM model_prices;"
