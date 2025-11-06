# =============================================
# ULTRA AI FOREX BOT – FINAL | RENDER WORKER
# NO PANDAS | NO NEST | 24/7 FREE
# =============================================

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import threading
import schedule

# ================= AUTO-RETRY =================
def retry(func):
    def wrapper(*args, **kwargs):
        for i in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Retry {i+1}/3: {e}")
                time.sleep(2 ** i)
        return None
    return wrapper

# ================= SETUP =================
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

MIN_CONFIDENCE = 97
MAX_SIGNALS = 1
COOLDOWN = {'scalping': 300, 'swing': 14400}

XAI_API_KEY = os.getenv("XAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID"))

# ================= GLOBALS =================
mode = None
pending = 0
last_signal = {s: 0 for s in SYMBOLS}
news_events = []

# ================= DATA =================
@retry
def get_klines(symbol, interval, limit=50):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': BINANCE_MAP[symbol], 'interval': interval, 'limit': limit}
    data = requests.get(url, params=params, timeout=10).json()
    return [{'close': float(c[4]), 'high': float(c[2]), 'low': float(c[3]), 'vol': float(c[5])} for c in data]

def ema(values, period):
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

# ================= SIGNAL LOGIC =================
def check_crossover(closes, fast=9, slow=21):
    if len(closes) < slow + 1: return None
    fast_ema = ema(closes[-fast:], fast)
    slow_ema = ema(closes[-slow:], slow)
    prev_fast = ema(closes[-fast-1:-1], fast)
    prev_slow = ema(closes[-slow-1:-1], slow)
    if prev_fast <= prev_slow and fast_ema > slow_ema:
        return 'buy'
    if prev_fast >= prev_slow and fast_ema < slow_ema:
        return 'sell'
    return None

def multi_tf_confirm(symbol, tf_list):
    signals = []
    for tf in tf_list:
        data = get_klines(symbol, tf)
        if not data: return None
        closes = [x['close'] for x in data]
        sig = check_crossover(closes)
        if sig: signals.append(sig)
    return signals[0] if len(signals) == len(tf_list) and len(set(signals)) == 1 else None

# ================= LLM CONFIDENCE =================
@retry
def grok_confidence(symbol, direction):
    prompt = f"Is {direction.upper()} on {symbol} 97%+ likely now? Reply JSON: {{'confidence': 0-100}}"
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            timeout=10
        )
        txt = resp.json()['choices'][0]['message']['content']
        conf = json.loads(txt.replace("```json","").replace("```","").strip())['confidence']
        return conf
    except Exception as e:
        print(f"LLM Error: {e}")
        return 50

# ================= ECON CALENDAR =================
@retry
def load_news():
    global news_events
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    data = requests.get(url, timeout=10).json()
    now = datetime.utcnow()
    news_events = []
    for e in data:
        if e.get('impact') == 'High':
            try:
                t = datetime.strptime(e['date'].split('T')[0] + ' ' + e['date'].split('T')[1].split('-')[0], "%Y-%m-%d %H:%M:%S")
                if t > now - timedelta(hours=1):
                    news_events.append(t)
            except:
                continue
    print(f"Calendar: {len(news_events)} events")

def news_block():
    now = datetime.utcnow()
    return any(abs((t - now).total_seconds()) < 1800 for t in news_events)

# ================= ALERT =================
async def send_signal(sym, dir, price):
    global pending
    if pending >= MAX_SIGNALS: return
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("YES – LOG", callback_data=f"yes|{sym}|{dir}")],
        [InlineKeyboardButton("NO – SKIP", callback_data=f"no|{sym}|{dir}")]
    ])
    msg = f"**ULTRA {mode.upper()}**\n`{dir.upper()} {sym}` @ `{price:.5f}`\n**LLM: 97%+**"
    await app.bot.send_message(ADMIN_USER_ID, msg, parse_mode='Markdown', reply_markup=kb)
    pending += 1

# ================= SCANNER =================
def scan():
    global mode, pending
    if not mode or pending or news_block(): return
    tfs = SCALP_TFS if mode == 'scalping' else SWING_TFS
    cd = COOLDOWN[mode]
    for sym in SYMBOLS:
        if time.time() - last_signal[sym] < cd: continue
        dir = multi_tf_confirm(sym, tfs)
        if not dir: continue
        data = get_klines(sym, '1m')
        if not data: continue
        price = data[-1]['close']
        conf = grok_confidence(sym, dir)
        if conf >= MIN_CONFIDENCE:
            asyncio.create_task(send_signal(sym, dir, price))
            last_signal[sym] = time.time()

# ================= TELEGRAM HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_USER_ID: return
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Scalping", callback_data="scalping")],
        [InlineKeyboardButton("Swing", callback_data="swing")]
    ])
    await update.message.reply_text("**ULTRA BOT LIVE**\nChoose mode:", reply_markup=kb, parse_mode='Markdown')

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global mode, pending
    q = update.callback_query
    await q.answer()
    if q.data in ['scalping', 'swing']:
        mode = q.data
        schedule.clear()
        schedule.every(60 if mode=='scalping' else 300).seconds.do(scan)
        await q.edit_message_text(f"**{mode.upper()} ON**")
    else:
        action = q.data.split('|')[0]
        await q.edit_message_text("**LOGGED**" if action == 'yes' else "Skipped")
        pending = 0

# ================= MAIN =================
async def main():
    global app
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button))

    load_news()
    schedule.every(6).hours.do(load_news)

    def run_scanner():
        while True:
            schedule.run_pending()
            time.sleep(1)
    threading.Thread(target=run_scanner, daemon=True).start()

    print("BOT READY – /start")
    await app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
