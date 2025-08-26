#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH AI Trading Bot — Pro MVP v2.1 (dziala)
=========================================
Wersja zawiera:
- Filtry reżimu rynku (trend EMA50/EMA200 + widełki zmienności ATR/Price)
- "Confidence buffer" dla progów prawdopodobieństwa
- Volatility targeting (skalowanie pozycji do target_vol annual)
- Kill switch na bazie drawdown (automatyczna pauza)
- Naprawki: import datetime, bezpieczniejsze alerty, pominięcie sygnałów "flat"

Uwaga: kod edukacyjny. Zanim użyjesz na realnym kapitale — testnet/paper + własne audyty.
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# third-party
import ccxt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import ta

# Optional: LLM explainability (graceful if missing)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# ----------------------------
# Logging
# ----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"), encoding="utf-8")
    ]
)
log = logging.getLogger("eth-bot")

# ----------------------------
# Config & constants
# ----------------------------
PAIR = "ETH/USDT"
DEFAULT_TIMEFRAME = "1h"
EXCHANGE_ID = "binance"
MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "meta.json")
EQUITY_PATH = os.path.join(LOG_DIR, "equity.csv")
KILL_PATH = os.path.join(LOG_DIR, "kill_switch.json")

@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01       # 1% of equity
    atr_period: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    min_qty: float = 0.0005
    fee_bps: float = 6.0              # 0.06% taker (oba kierunki ~12 bps round-trip)
    slip_bps: float = 5.0             # 0.05% slippage est.

@dataclass
class TrainConfig:
    years: int = 2
    test_splits: int = 5

@dataclass
class StrategyConfig:
    confidence_buffer: float = 0.05      # ile ponad próg musi być proba
    use_trend_filter: bool = True        # EMA50 vs EMA200
    vol_perc_min: float = 0.005          # 0.5% minimalny ATR/Price
    vol_perc_max: float = 0.05           # 5% maksymalny ATR/Price
    target_vol_annual: float = 0.20      # 20% rocznie
    dd_threshold: float = 0.10           # 10% equity DD -> pauza
    dd_pause_hours: int = 24             # pauza 24h

STRAT = StrategyConfig()

# ----------------------------
# Utility
# ----------------------------

def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str, default: Optional[Dict] = None) -> Dict:
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Data
# ----------------------------

def get_exchange(paper: bool = False) -> ccxt.Exchange:
    """Create ccxt exchange (Binance). If paper=True, connect to futures testnet for liquidity."""
    ex_class = getattr(ccxt, EXCHANGE_ID)
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if paper:
        params.update({
            "urls": {
                "api": {
                    "public": "https://testnet.binancefuture.com/fapi/v1",
                    "private": "https://testnet.binancefuture.com/fapi/v1",
                }
            }
        })
    ex = ex_class(params)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if api_key and api_secret:
        ex.apiKey = api_key
        ex.secret = api_secret
    return ex


def fetch_ohlcv(exchange: ccxt.Exchange, pair: str, timeframe: str, since_ms: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    all_rows: List[List[float]] = []
    last_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=last_since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        last_since = batch[-1][0]
        time.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        raise RuntimeError("No OHLCV fetched.")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df

# ----------------------------
# Features & targets
# ----------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMAs / trend
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()

    # Momentum
    out["rsi_14"] = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()
    macd = ta.trend.MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"], out["macd_signal"] = macd.macd(), macd.macd_signal()
    stoch = ta.momentum.StochasticOscillator(high=out["high"], low=out["low"], close=out["close"], window=14, smooth_window=3)
    out["stoch_k"], out["stoch_d"] = stoch.stoch(), stoch.stoch_signal()
    out["willr"] = ta.momentum.WilliamsRIndicator(high=out["high"], low=out["low"], close=out["close"], lbp=14).williams_r()
    out["roc_10"] = ta.momentum.ROCIndicator(close=out["close"], window=10).roc()

    # Volatility / bands
    bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_high"], out["bb_low"], out["bb_bw"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=14)
    out["atr_14"] = atr.average_true_range()

    # Lags & returns
    out["ret_1"] = out["close"].pct_change()
    out["ret_3"] = out["close"].pct_change(3)
    out["vol_1"] = out["volume"].pct_change()

    # Target (binary): next close up or down
    out["ret_fwd_1"] = out["close"].shift(-1) / out["close"] - 1.0
    out["y"] = (out["ret_fwd_1"] > 0).astype(int)

    return out.dropna()


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df[[
        "close","volume",
        "ema_12","ema_20","ema_50","ema_200",
        "rsi_14","macd","macd_signal","stoch_k","stoch_d","willr","roc_10",
        "bb_high","bb_low","bb_bw","atr_14",
        "ret_1","ret_3","vol_1"
    ]]
    target = df["y"].astype(int)
    return features, target

# ----------------------------
# Modeling & walk-forward backtest (bez zmian istotowych)
# ----------------------------

def build_model() -> Pipeline:
    """Soft-voting ensemble with calibrated probabilities."""
    gb = GradientBoostingClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    base = VotingClassifier(estimators=[("gb", gb), ("rf", rf)], voting="soft")
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", calibrated)
    ])
    return pipe


def walk_forward_backtest(X: pd.DataFrame, y: pd.Series, n_splits: int, fee_bps: float, slip_bps: float) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1s, precs, recs = [], [], [], []
    best_thresholds = []
    pipeline = None

    for _, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = build_model()
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        grid = np.linspace(0.45, 0.65, 21)  # możesz rozszerzyć
        best_f1, best_thr = -1.0, 0.5
        for thr in grid:
            pred = (proba >= thr + STRAT.confidence_buffer).astype(int)
            f1v = f1_score(y_test, pred)
            # kara za flipy (koszty)
            flips = np.sum(np.abs(np.diff(pred)))
            round_trip_bps = fee_bps * 2 + slip_bps * 2
            cost_penalty = flips * (round_trip_bps / 10000.0) / max(1, len(y_test))
            f1v = max(0.0, f1v - cost_penalty)
            if f1v > best_f1:
                best_f1, best_thr = f1v, thr
        accs.append(accuracy_score(y_test, (proba >= best_thr).astype(int)))
        f1s.append(best_f1)
        precs.append(precision_score(y_test, (proba >= best_thr).astype(int)))
        recs.append(recall_score(y_test, (proba >= best_thr).astype(int)))
        best_thresholds.append(best_thr)

    return {
        "acc_mean": float(np.mean(accs)),
        "f1_mean": float(np.mean(f1s)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "best_threshold": float(np.median(best_thresholds)),
        "last_pipeline": pipeline,
    }

# ----------------------------
# Helpers: regime, vol targeting, equity/DD
# ----------------------------

def timeframe_to_hours(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return int(tf[:-1]) * 1.0
    if tf.endswith("d"):
        return int(tf[:-1]) * 24.0
    return 1.0  # default 1h


def compute_regime(df_feat: pd.DataFrame) -> Dict[str, float | bool]:
    last = df_feat.iloc[-1]
    trend_up = bool(last["ema_50"] > last["ema_200"])  # True=UP, False=DOWN
    vol_perc = float(last["atr_14"] / max(1e-9, last["close"]))
    vol_ok = (vol_perc >= STRAT.vol_perc_min) and (vol_perc <= STRAT.vol_perc_max)
    return {"trend_up": trend_up, "vol_perc": vol_perc, "vol_ok": vol_ok}


def realized_vol_annual(df_feat: pd.DataFrame, timeframe: str) -> float:
    # prosta estymacja annualized realized vol z 20 okien
    ret = df_feat["ret_1"].dropna().iloc[-20:]
    if len(ret) < 5:
        return 0.0
    hourly = ret.std() * math.sqrt(1.0 / max(1e-9, timeframe_to_hours(timeframe)))
    return float(hourly * math.sqrt(24 * 365))


def calc_position_size(equity_usdt: float, entry: float, atr: float, risk: RiskConfig, rv_annual: float) -> float:
    if atr <= 0 or equity_usdt <= 0:
        return 0.0
    stop_distance = risk.atr_mult_sl * atr
    risk_amount = equity_usdt * risk.risk_per_trade
    qty = risk_amount / stop_distance
    # Vol targeting — skaluje w dół przy wysokiej realized vol
    if rv_annual > 0:
        scaler = min(1.0, STRAT.target_vol_annual / rv_annual)
        qty *= scaler
    return max(qty, 0.0)


def build_bracket_prices(side: str, entry: float, atr: float, risk: RiskConfig) -> Tuple[float, float]:
    if side == "buy":
        sl = entry - risk.atr_mult_sl * atr
        tp = entry + risk.atr_mult_tp * atr
    elif side == "sell":
        sl = entry + risk.atr_mult_sl * atr
        tp = entry - risk.atr_mult_tp * atr
    else:
        sl, tp = entry, entry
    return sl, tp

# ----------------------------
# Alerts
# ----------------------------
import urllib.request

ALERTS_FILE = os.path.join(LOG_DIR, "alerts.csv")
os.makedirs(LOG_DIR, exist_ok=True)

def log_alert(signal: str, price: float, proba: float):
    row = {
        "timestamp": datetime.utcnow(),
        "signal": signal,
        "price": price,
        "proba": proba
    }
    if os.path.exists(ALERTS_FILE):
        df = pd.read_csv(ALERTS_FILE, parse_dates=["timestamp"])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(ALERTS_FILE, index=False)

def _mask_webhook(url: str) -> str:
    if not url:
        return "<empty>"
    return url[:40] + "/.../" + url.split("/")[-1][-6:]


def send_discord(msg: str) -> None:
    url = (os.getenv("DISCORD_WEBHOOK_URL") or "").strip().strip('"').strip("'")
    if not url or not url.startswith("https://discord.com/api/webhooks/"):
        log.warning("Discord alert skipped: invalid/missing webhook URL.")
        return
    try:
        try:
            import requests
        except ImportError:
            requests = None
        payload = {"content": msg}
        if requests:
            r = requests.post(url, json=payload, timeout=10, headers={"Content-Type": "application/json"})
            if r.status_code != 204:
                log.warning(f"Discord alert failed: {r.status_code} {r.text} (url={_mask_webhook(url)})")
        else:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as _:
                    pass
            except Exception as e:
                body = ""
                if hasattr(e, "read"):
                    try:
                        body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                log.warning(f"Discord alert failed: {e} {body} (url={_mask_webhook(url)})")
    except Exception as e:
        log.warning(f"Discord alert unexpected error: {e} (url={_mask_webhook(url)})")


def send_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": msg}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
    except Exception as e:
        log.warning(f"Telegram alert failed: {e}")


def alert(msg: str, channels: List[str]) -> None:
    log.info(msg)
    if "discord" in channels:
        send_discord(msg)
    if "telegram" in channels:
        send_telegram(msg)

# ----------------------------
# LLM explanation (optional)
# ----------------------------

def llm_explain(decision: dict) -> Optional[str]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key)
        msg = (
            "You are a cautious crypto trading assistant. Summarize the signal below in 3-4 bullet points, "
            "include trend, momentum, volatility, and the proposed risk controls. Avoid hype; include a reminder about risk."
            f"Signal JSON: {json.dumps(decision, default=str)}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": msg}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM explanation unavailable: {e}"

# ----------------------------
# Equity & Kill switch
# ----------------------------

def fetch_equity_usdt(ex: ccxt.Exchange, fallback: float) -> float:
    try:
        bal = ex.fetch_balance()
        # Binance futures USDT wallet
        total = bal.get("total") or {}
        eq = float(total.get("USDT") or 0.0)
        if eq > 0:
            return eq
    except Exception:
        pass
    return float(fallback)


def record_equity(value: float) -> None:
    row = {"timestamp": datetime.utcnow(), "equity": float(value)}
    if os.path.exists(EQUITY_PATH):
        df = pd.read_csv(EQUITY_PATH, parse_dates=["timestamp"])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EQUITY_PATH, index=False)


def current_drawdown() -> Tuple[float, float, float]:
    if not os.path.exists(EQUITY_PATH):
        return 0.0, 0.0, 0.0
    df = pd.read_csv(EQUITY_PATH, parse_dates=["timestamp"])
    eq = df["equity"].values
    peak = float(np.max(eq))
    last = float(eq[-1])
    dd = 0.0 if peak <= 0 else (peak - last) / peak
    return last, peak, dd


def is_paused() -> bool:
    data = load_json(KILL_PATH, default={})
    until = data.get("paused_until")
    if not until:
        return False
    try:
        return datetime.utcnow() < datetime.fromisoformat(until)
    except Exception:
        return False


def trigger_pause(hours: int) -> None:
    until = datetime.utcnow() + timedelta(hours=hours)
    save_json(KILL_PATH, {"paused_until": until.isoformat()})

# ----------------------------
# Signal computation with filters
# ----------------------------

def compute_signal(df_feat: pd.DataFrame, pipeline: Pipeline, threshold: float, timeframe: str) -> Dict:
    X, _ = feature_target_split(df_feat)
    latest_row = X.iloc[-1:]
    proba_up = float(pipeline.predict_proba(latest_row)[0][1])

    last = df_feat.iloc[-1]
    regime = compute_regime(df_feat)
    conf_thr = threshold + STRAT.confidence_buffer

    side = "flat"
    # Trend + confidence gating
    if STRAT.use_trend_filter:
        if regime["vol_ok"]:
            if regime["trend_up"] and proba_up >= conf_thr:
                side = "buy"
            elif (not regime["trend_up"]) and proba_up <= (1.0 - conf_thr):
                side = "sell"
    else:
        if regime["vol_ok"]:
            if proba_up >= conf_thr:
                side = "buy"
            elif proba_up <= (1.0 - conf_thr):
                side = "sell"

    rv = realized_vol_annual(df_feat, timeframe)

    return {
        "price": float(last["close"]),
        "atr": float(last["atr_14"]),
        "proba_up": proba_up,
        "side": side,
        "t": str(df_feat.index[-1]),
        "trend_up": bool(regime["trend_up"]),
        "vol_perc": float(regime["vol_perc"]),
        "vol_ok": bool(regime["vol_ok"]),
        "rv_annual": float(rv),
    }

# ----------------------------
# Modes
# ----------------------------

def mode_backtest(args: argparse.Namespace) -> None:
    load_dotenv()
    log.info("Starting backtest…")
    ex = get_exchange(paper=False)

    now_ms = int(time.time() * 1000)
    ms_per_year = 365 * 24 * 60 * 60 * 1000
    since_ms = now_ms - args.years * ms_per_year

    df = fetch_ohlcv(ex, PAIR, args.timeframe, since_ms=since_ms)
    df_feat = add_indicators(df)
    X, y = feature_target_split(df_feat)

    risk = RiskConfig()
    results = walk_forward_backtest(X, y, n_splits=args.splits, fee_bps=risk.fee_bps, slip_bps=risk.slip_bps)

    log.info(
        "Walk-forward — acc: %.3f, f1: %.3f, precision: %.3f, recall: %.3f, thr*: %.3f",
        results['acc_mean'], results['f1_mean'], results['precision_mean'], results['recall_mean'], results['best_threshold']
    )

    # Save model & meta
    import pickle
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(results["last_pipeline"], f)
    meta = {"best_threshold": results["best_threshold"], "timeframe": args.timeframe}
    save_json(META_PATH, meta)
    log.info("Saved model -> %s and meta -> %s", os.path.join(MODEL_DIR, "ensemble.pkl"), META_PATH)


def load_model_and_meta() -> Tuple[Pipeline, float, str]:
    import pickle
    model_path = os.path.join(MODEL_DIR, "ensemble.pkl")
    if not os.path.exists(model_path):
        log.error("Model not found. Run backtest first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        pipeline: Pipeline = pickle.load(f)
    meta = load_json(META_PATH, default={})
    threshold = float(meta.get("best_threshold", 0.55))
    timeframe = str(meta.get("timeframe", DEFAULT_TIMEFRAME))
    return pipeline, threshold, timeframe


def mode_daemon(args: argparse.Namespace) -> None:
    load_dotenv()
    log.info("Starting daemon… (24/7)")

    channels = [c.strip().lower() for c in (args.alerts or [])]

    ex = get_exchange(paper=True) if args.paper or args.execute else get_exchange(paper=False)
    pipeline, threshold, tf_from_meta = load_model_and_meta()
    timeframe = args.timeframe or tf_from_meta

    # Inicjalna equity (fallback 10k), potem spróbuj pobrać z giełdy
    equity = 10_000.0
    equity = fetch_equity_usdt(ex, equity)
    record_equity(equity)

    risk = RiskConfig()
    last_decision: Optional[Dict] = None

    while True:
        try:
            # Kill switch
            if is_paused():
                log.warning("⚠️ Trading paused by kill switch — waiting…")
                time.sleep(max(30, int(args.poll)))
                continue

            now_ms = int(time.time() * 1000)
            since_ms = now_ms - 200 * 24 * 60 * 60 * 1000
            df = fetch_ohlcv(ex, PAIR, timeframe, since_ms=since_ms)
            df_feat = add_indicators(df)

            sig = compute_signal(df_feat, pipeline, threshold, timeframe)
            side = sig["side"]

            # jeśli flat — tylko zaloguj i omiń
            if side == "flat":
                log.info(
                    f"⏸ FLAT — trend_up={sig['trend_up']} vol_ok={sig['vol_ok']} vol%={sig['vol_perc']*100:.2f} p(up)={sig['proba_up']:.3f}"
                )
                time.sleep(max(5, int(args.poll)))
                continue

            sl, tp = build_bracket_prices(side, sig["price"], sig["atr"], risk)
            rv = sig["rv_annual"]
            qty = calc_position_size(equity, sig["price"], sig["atr"], risk, rv)
            qty = max(qty, risk.min_qty)

            decision = {
                "pair": PAIR,
                "timeframe": timeframe,
                "price": sig["price"],
                "atr": sig["atr"],
                "proba_up": sig["proba_up"],
                "threshold": threshold,
                "conf_buffer": STRAT.confidence_buffer,
                "side": side,
                "qty": float(qty),
                "stop": float(sl),
                "take": float(tp),
                "t": sig["t"],
                "trend_up": sig["trend_up"],
                "vol_perc": sig["vol_perc"],
                "rv_annual": rv,
            }

            should_send = (
                last_decision is None
                or decision["side"] != last_decision.get("side")
                or abs(decision["price"] - last_decision.get("price", decision["price"])) > 0.1
                or abs(decision["proba_up"] - last_decision.get("proba_up", decision["proba_up"])) > 0.01
            )

            alert_sent = False
            if should_send:
                title = f"[ETH BOT] {decision['t']}"
                description = (
                    f"Price: {decision['price']:.2f} | P(up): {decision['proba_up']:.3f} (thr {decision['threshold']:.2f}+buf {decision['conf_buffer']:.2f})\n"
                    f"Action: {decision['side'].upper()}  Qty≈{decision['qty']:.5f}\n"
                    f"TP: {decision['take']:.2f} | SL: {decision['stop']:.2f} | vol%: {decision['vol_perc']*100:.2f} | RV_annual≈{decision['rv_annual']:.2f}"
                )
                txt = f"{title}\n{description}"
                alert(txt, channels)
                last_decision = decision
                alert_sent = True

            if args.explain and alert_sent:
                expl = llm_explain(decision)
                if expl:
                    alert("LLM explain:\n" + expl, channels)

            # Wykonanie + kill switch na DD
            if args.execute:
                try:
                    order = ex.create_order(symbol=PAIR, type="market", side=side, amount=qty)
                    log.info("Entry order placed: %s", order.get('id'))
                    if side == "buy":
                        ex.create_order(symbol=PAIR, type="stop_market", side="sell", amount=qty, params={"stopPrice": decision['stop'], "reduceOnly": True})
                        ex.create_order(symbol=PAIR, type="take_profit_market", side="sell", amount=qty, params={"stopPrice": decision['take'], "reduceOnly": True})
                    else:
                        ex.create_order(symbol=PAIR, type="stop_market", side="buy", amount=qty, params={"stopPrice": decision['stop'], "reduceOnly": True})
                        ex.create_order(symbol=PAIR, type="take_profit_market", side="buy", amount=qty, params={"stopPrice": decision['take'], "reduceOnly": True})
                    log.info("Bracket orders placed (TP/SL)")
                except Exception as e:
                    log.error("Order placement failed: %s", e)

            # Aktualizacja equity i test DD
            equity = fetch_equity_usdt(ex, equity)
            record_equity(equity)
            last, peak, dd = current_drawdown()
            if dd >= STRAT.dd_threshold:
                msg = f"🛑 Kill switch: DD={dd*100:.2f}% (peak={peak:.2f}, last={last:.2f}) — pause {STRAT.dd_pause_hours}h"
                alert(msg, channels)
                trigger_pause(STRAT.dd_pause_hours)

            time.sleep(max(5, int(args.poll)))
        except KeyboardInterrupt:
            log.info("Daemon stopped by user.")
            break
        except Exception as e:
            log.error("Daemon loop error: %s", e)
            time.sleep(10)

# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETH AI Trading Bot — Pro MVP v2.1")
    sub = p.add_subparsers(dest="mode", required=True)

    # Backtest
    b = sub.add_parser("backtest", help="Train & backtest model with walk-forward evaluation")
    b.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, type=str)
    b.add_argument("--years", default=2, type=int)
    b.add_argument("--splits", default=5, type=int)
    b.set_defaults(func=mode_backtest)

    # Daemon (24/7)
    d = sub.add_parser("daemon", help="24/7 analysis + alerts (+ optional orders)")
    d.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, type=str)
    d.add_argument("--poll", default=60, type=int, help="seconds between runs")
    d.add_argument("--alerts", nargs="*", default=[], help="alert channels: discord telegram")
    d.add_argument("--paper", action="store_true", help="use testnet endpoints")
    d.add_argument("--execute", action="store_true", help="place orders as well")
    d.add_argument("--explain", action="store_true", help="ask LLM to explain decision")
    d.set_defaults(func=mode_daemon)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
