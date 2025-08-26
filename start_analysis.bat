@echo off
cd /d %~dp0
call venv\Scripts\activate
python eth_ai_trading_bot.py daemon --timeframe 1h --poll 60 --alerts discord
pause