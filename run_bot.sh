#!/bin/bash

PROJECT_DIR="/home/username/eth_trading_bot"
PYTHON="/usr/bin/python3"
RESTART_INTERVAL=5400  # 1,5 godziny w sekundach

cd $PROJECT_DIR

while true
do
    echo "[$(date)] 🔹 START: Trenowanie modelu + analiza..."

    # Uruchamiamy proces w tle i mierzymy czas
    timeout $RESTART_INTERVAL $PYTHON eth_ai_trading_bot.py backtest --timeframe 1h --years 2 --splits 5
    timeout $RESTART_INTERVAL $PYTHON eth_ai_trading_bot.py daemon --timeframe 1h --poll 60 --alerts discord

    echo "[$(date)] 🔄 Restart bota po 1,5 godziny..."
done
