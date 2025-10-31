# =============================================
# ULTRA AI FOREX BOT – FINAL (NO PANDAS-TA)
# 6 Pairs | Scalping/Swing | Multi-TF + Econ Filter + Auto-Retry
# 97%+ Confidence | Telegram Alerts | Render-Ready
# =============================================

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import schedule
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import threading
import nest_asyncio
import functools

# ================= AUTO-RETRY DECORATOR =================
def retry(max_attempts=3, delay=1, backoff=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"{func.__name__} FAILED after {max_attempts} attempts: {e}")
                        return None
                    print(f"{func.__name__} error: {e} | Retry {attempts}/{max_attempts} in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
nest_asyncio.apply()

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'NZDUSD', 'XAUUSD', 'NAS100']
BINANCE_MAP = {
    'EURUSD': 'EURUSDT', 'GBPUSD': 'GBPUSDT', 'USDJPY': 'USDJPY',
    'NZDUSD': 'NZDUSDT', 'XAUUSD': 'XAUUSDT', 'NAS100': 'NAS100USD'
}

SCALP_TFS = ['1m', '5m', '15m']
SWING_TFS = ['1h', '4h']

RISK_PCT = 0.01
RR_RATIO = 3.0
MIN_STRENGTH = 97
SL_PIPS_SCALP = 12
SL_PIPS_SWING = 45
MAX_ACTIVE_SIGNALS = 1
SCALP_COOLDOWN = 300
SWING_COOLDOWN = 14400

XAI_API_KEY = os.getenv("XAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID"))

# ================= GLOBAL STATE =================
current_mode = None
scaler = StandardScaler()
xgb_model = xgb.XGBClassifier()
sentiment_pipeline = None
application = None
pending_signals = 0
last_signal_time = {sym: 0 for sym in SYMBOLS}
news_events = []
econ_blocked_until = 0

# ================= MODEL =================
def train_ultra_model():
    print("Training ULTRA model...")
    np.random.seed(42)
    n = 10000
    X = np.random.rand(n, 9)
    ema_bull = (X[:, 0] > X[:, 1]) & (X[:, 2] > 0.55) & (X[:, 2] < 0.65)
    ema_bear = (X[:, 0] < X[:, 1]) & (X[:, 2] < 0.45) & (X[:, 2] > 0.35)
    vol_ok = X[:, 8] > 0.3
    y = np.where(ema_bull & vol_ok, 1, np.where(ema_bear & vol_ok, 0, -1))
    y = (y == 1).astype(int)
    X_scaled = scaler.fit_transform(X)
    xgb_model.fit(X_scaled, y)
    xgb_model.save_model("ultra_xgb_model.json")
    joblib.dump(scaler, "ultra_scaler.pkl")
    print("Model trained.")

def load_models():
    global xgb_model, scaler, sentiment_pipeline
    if os.path.exists("ultra_xgb_model.json") and os.path.exists("ultra_scaler.pkl"):
        try:
            xgb_model.load_model("ultra_xgb_model.json")
            scaler = joblib.load("ultra_scaler.pkl")
            print("Model loaded.")
        except:
            train_ultra_model()
    else:
        train_ultra_model()

    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("FinBERT loaded.")
    except Exception as e:
        print(f"FinBERT failed: {e}")

load_models()

# ================= BINANCE DATA =================
@retry()
def get_ohlcv(symbol, interval, limit=200):
    binance_sym = BINANCE_MAP.get(symbol, symbol)
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={'symbol': binance_sym, 'interval': interval, 'limit': limit}, timeout=10)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")
    data = resp.json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'a','b','c','d','e','f'])
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# ================= MANUAL INDICATORS (NO PANDAS-TA) =================
def add_technical_features(df):
    # EMA
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    
    return df.fillna(0)

# ================= ML + NLP + LLM =================
def ml_predict(df):
    latest = df.iloc[-1:]
    features = ['ema9', 'ema21', 'rsi', 'atr', 'bb_upper', 'bb_lower', 'volatility']
    X = latest[features]
    X_scaled = scaler.transform(X)
    prob = xgb_model.predict_proba(X_scaled)[0][1]
    return prob * 100

@retry()
def get_sentiment_score(symbol):
    query = symbol[:3] if 'USD' in symbol else 'gold' if 'XAU' in symbol else 'nasdaq'
    url = f"https://newsapi.org/v2/everything?q={query}+market&apiKey={NEWS_API_KEY}&pageSize=5"
    resp = requests.get(url, timeout=10).json()
    articles = [a['title'] + " " + a.get('description','') for a in resp.get('articles', [])]
    if not articles: return 50
    results = sentiment_pipeline(articles)
    pos = sum(1 for r in results if r['label'] == 'Positive')
    return (pos / len(results)) * 100

@retry()
def llm_refine_signal(symbol, signal, ml_score, sentiment_score, tf_name):
    prompt = f"""
    Only approve if {signal.upper()} on {symbol} ({tf_name}) is 97%+ likely.
    ML: {ml_score:.1f}%, Sentiment: {sentiment_score:.1f}%.
    Return JSON: {{ "approve": true/false, "confidence": 0-100, "reason": "<30 chars>" }}
    """
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
        timeout=15
    )
    text = resp.json()['choices'][0]['message']['content']
    result = json.loads(text.replace("```json","").replace("```","").strip())
    return result.get('approve', False), result.get('confidence', 50), result.get('reason', 'No reason')

def final_strength(ml_score, sentiment_score, llm_confidence):
    return min(ml_score * 0.35 + sentiment_score * 0.25 + llm_confidence * 0.40, 100)

# ================= ECONOMIC CALENDAR =================
@retry()
def fetch_econ_calendar():
    global news_events
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    now = datetime.utcnow()
    high_impact = []
    for event in data:
        if event.get('impact') == 'High':
            try:
                event_time = datetime.strptime(event['date'], "%Y-%m-%d %H:%M:%S")
                if event_time > now - timedelta(hours=1):
                    high_impact.append(event_time)
            except:
                continue
    news_events = high_impact
    print(f"Econ calendar loaded: {len(high_impact)} high-impact events.")

def is_news_blocked():
    global econ_blocked_until
    now = datetime.utcnow()
    if now < econ_blocked_until:
        return True, econ_blocked_until
    for event in news_events:
        if abs((event - now).total_seconds()) < 1800:
            econ_blocked_until = event + timedelta(minutes=30)
            return True, econ_blocked_until
    return False, None

# ================= MULTI-TF CONFIRMATION =================
def multi_tf_confirm(symbol, mode):
    tfs = SCALP_TFS if mode == 'scalping' else SWING_TFS
    signals = []
    for tf in tfs:
        df = get_ohlcv(symbol, tf)
        if df is None or len(df) < 50: return None
        df = add_technical_features(df)
        sig = generate_signal(df, mode)
        if sig:
            signals.append(sig)
    if len(signals) == len(tfs) and all(s == signals[0] for s in signals):
        return signals[0]
    return None

# ================= SIGNAL LOGIC =================
def generate_signal(df, mode):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if mode == 'scalping':
        if (prev['ema9'] <= prev['ema21'] and latest['ema9'] > latest['ema21'] and 
            55 < latest['rsi'] < 65 and latest['volatility'] > 0.0008):
            return 'buy'
        if (prev['ema9'] >= prev['ema21'] and latest['ema9'] < latest['ema21'] and 
            35 < latest['rsi'] < 45 and latest['volatility'] > 0.0008):
            return 'sell'
    else:
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['close']
        if (prev['ema9'] <= prev['ema21'] and latest['ema9'] > latest['ema21'] and 
            bb_width > 0.018 and latest['rsi'] > 60):
            return 'buy'
        if (prev['ema9'] >= prev['ema21'] and latest['ema9'] < latest['ema21'] and 
            bb_width > 0.018 and latest['rsi'] < 40):
            return 'sell'
    return None

# ================= SEND SIGNAL =================
async def send_trade_alert(symbol, signal, entry, sl, tp, strength, reason):
    global pending_signals
    if pending_signals >= MAX_ACTIVE_SIGNALS: return
    keyboard = [
        [InlineKeyboardButton("YES - Log", callback_data=f"yes|{symbol}|{signal}")],
        [InlineKeyboardButton("NO - Skip", callback_data=f"no|{symbol}|{signal}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    msg = (f"**ULTRA {current_mode.upper()} SIGNAL**\n"
           f"`{signal.upper()} {symbol}`\n"
           f"Entry: `{entry:.5f}` | SL: `{sl:.5f}` | TP: `{tp:.5f}`\n"
           f"Strength: `{strength:.1f}%` | {reason}\n"
           f"**MULTI-TF + ECON SAFE**")
    await application.bot.send_message(
        chat_id=ADMIN_USER_ID,
        text=msg,
        parse_mode='Markdown',
        reply_markup=reply_markup
    )
    pending_signals += 1

# ================= CALLBACKS =================
async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_mode
    query = update.callback_query
    await query.answer()
    mode = query.data
    current_mode = mode
    schedule.clear()
    interval = 60 if mode == 'scalping' else 300
    schedule.every(interval).seconds.do(scan_once)
    await query.edit_message_text(
        f"**{mode.upper()} MODE ACTIVATED**\n"
        f"Pairs: 6 | Cooldown: { (SCALP_COOLDOWN if mode == 'scalping' else SWING_COOLDOWN)//60 } min\n"
        f"Signals: **≥97% only**\n"
        f"First scan in 60s...",
        parse_mode='Markdown'
    )

async def trade_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global pending_signals
    query = update.callback_query
    await query.answer()
    data = query.data.split('|')
    if data[0] == 'yes':
        await query.edit_message_text("**CONFIRMED & LOGGED**", parse_mode='Markdown')
        with open("ultra_log.txt", "a") as f:
            f.write(f"{datetime.now()} | CONFIRMED | {data[2].upper()} {data[1]}\n")
    else:
        await query.edit_message_text("Signal skipped.")
    pending_signals -= 1

# ================= COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID: return
    keyboard = [
        [InlineKeyboardButton("Scalping", callback_data="scalping")],
        [InlineKeyboardButton("Swing", callback_data="swing")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "**ULTRA AI BOT**\n"
        "Select Mode:\n"
        "• Scalping: 1m/5m/15m | 5 min cooldown\n"
        "• Swing: 1h | 4h cooldown\n"
        "Pairs: `EURUSD, GBPUSD, USDJPY, NZDUSD, XAUUSD, NAS100`",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

# ================= SCANNER =================
def scan_once():
    global pending_signals, last_signal_time, current_mode
    if not current_mode or pending_signals >= MAX_ACTIVE_SIGNALS: return

    blocked, until = is_news_blocked()
    if blocked:
        print(f"ECON BLOCKED until {until}")
        return

    cooldown = SCALP_COOLDOWN if current_mode == 'scalping' else SWING_COOLDOWN
    for symbol in SYMBOLS:
        now = time.time()
        if now - last_signal_time[symbol] < cooldown: continue

        signal = multi_tf_confirm(symbol, current_mode)
        if not signal: continue

        df = get_ohlcv(symbol, SCALP_TFS[0] if current_mode == 'scalping' else SWING_TFS[0])
        if df is None: continue
        df = add_technical_features(df)

        ml_score = ml_predict(df)
        sentiment_score = get_sentiment_score(symbol)
        approve, llm_conf, reason = llm_refine_signal(symbol, signal, ml_score, sentiment_score, "multi")
        if not approve: continue

        strength = final_strength(ml_score, sentiment_score, llm_conf)
        if strength < MIN_STRENGTH: continue

        entry = df['close'].iloc[-1]
        point = 0.0001 if 'JPY' in symbol else 0.01
        if symbol in ['XAUUSD', 'NAS100']: point = 0.1
        sl_pips = SL_PIPS_SCALP if current_mode == 'scalping' else SL_PIPS_SWING
        sl_distance = sl_pips * point
        tp_distance = sl_distance * RR_RATIO
        sl = entry - sl_distance if signal == 'buy' else entry + sl_distance
        tp = entry + tp_distance if signal == 'buy' else entry - tp_distance

        asyncio.create_task(send_trade_alert(symbol, signal, entry, sl, tp, strength, reason))
        last_signal_time[symbol] = now
        time.sleep(3)

# ================= MAIN =================
async def main():
    global application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern=r'^(scalping|swing)$'))
    application.add_handler(CallbackQueryHandler(trade_callback, pattern=r'^(yes|no)\|'))

    fetch_econ_calendar()
    schedule.every(6).hours.do(fetch_econ_calendar)

    def run_scanner():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=run_scanner, daemon=True).start()
    print("ULTRA BOT READY – SEND /start")
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
