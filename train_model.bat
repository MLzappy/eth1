@echo off
cd /d %~dp0
call venv\Scripts\activate
python eth_ai_trading_bot.py backtest --timeframe 1h --years 2 --splits 5
pause