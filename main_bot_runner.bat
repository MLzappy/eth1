@echo off
title ETH AI Trading Bot - Main Runner
echo [BOT] Uruchamianie glownego procesu...

:loop
    echo.
    echo [BOT] Trenowanie modelu...
    call train_model.bat

    echo.
    echo [BOT] Start analizy...
    start "" /wait start_analysis.bat

    echo.
    echo [BOT] Restart analizy po 1.5 godziny (5400 sekund)...
    timeout /t 5400 >nul

    echo.
    echo [BOT] Restartuję bota...
goto loop
