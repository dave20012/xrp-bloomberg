@echo off
echo Link your Railway project and configure services:
echo railway link
echo railway service up --name web
echo railway service up --name inflow-worker
echo railway service up --name analytics-worker
echo railway service up --name news-worker
echo Set environment variables BINANCE_API_KEY BINANCE_API_SECRET NEWS_API_KEY HF_TOKEN CRYPTOCOMPARE_API_KEY DEEPSEEK_API_KEY DATABASE_URL REDIS_URL via Railway dashboard or CLI.
