# XRP Bloomberg-Style Intelligence Terminal

A production-ready, volume- and flow-centric XRP analytics terminal with derivatives context, anomaly detection, and regulatory/news overlays. Built with Streamlit, PostgreSQL, and Redis; deployable on Railway.

## Features
- Streamlit dark dashboard summarizing flow pressure, derivatives posture, anomaly detection, accumulation/distribution, manipulation heuristics, regulatory/news overlays, and a composite risk/health score.
- Geometry + swarm predictive stack that treats market state as a latent field and aggregates many cheap agents into a consensus.
- Workers for data ingestion, signal computation, news tagging, state vector construction, geometry projection, and swarm aggregation with Redis caching and PostgreSQL persistence.
- Deterministic offline fallbacks for environments without immediate API connectivity.

## Environment Variables
Configure via `.env` (not committed) or platform settings:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `NEWS_API_KEY`
- `HF_TOKEN`
- `CRYPTOCOMPARE_API_KEY`
- `DEEPSEEK_API_KEY`
- `DATABASE_URL` (PostgreSQL URL only)
- `REDIS_URL` (Redis URL only)

See `.env.example` for the template.

## Local Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # populate with your values
```

## Running the Dashboard
```bash
streamlit run main.py --server.port 8080 --server.address 0.0.0.0
```

## Workers
Each worker can run once or loop with an interval:
```bash
python workers/inflow_worker.py --loop --interval 300
python workers/analytics_worker.py --loop --interval 600
python workers/news_worker.py --loop --interval 1800
python workers/state_worker.py --loop --interval 300
python workers/geometry_worker.py --loop --interval 600
python workers/swarm_worker.py --loop --interval 600
```

## Scheduler
```bash
python workers/scheduler.py --loop --interval 900
```

## Railway Deployment
Scaffold services with the provided scripts:
```bash
bash railway_setup.sh
# or on Windows
railway_setup.bat
```

Recommended Railway commands:
```bash
railway link
railway service up --name web --command "streamlit run main.py --server.port 8080 --server.address 0.0.0.0"
railway service up --name inflow-worker --command "python workers/inflow_worker.py --loop"
railway service up --name analytics-worker --command "python workers/analytics_worker.py --loop"
railway service up --name news-worker --command "python workers/news_worker.py --loop"
```

Ensure all environment variables above are set in Railway.

## Testing
```bash
pytest
```

## Data Model
- `flows`: exchange inflow/outflow snapshots.
- `ohlcv`: hourly OHLCV bars.
- `openinterest`: derivatives open interest.
- `scores`: composite and component scores.
- `news`: tagged headlines.
- `market_state_snapshots`: normalized feature vectors and composite axes.
- `geometry_snapshots`: projected coordinates, motifs, and drift vectors.
- `swarm_snapshots`: aggregated swarm scores and contributing agents.

## Security Note
Environment variables are required for real API calls. No secrets are stored in the repository; placeholders are used throughout the codebase.
