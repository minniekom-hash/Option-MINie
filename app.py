"""OPTION TRADE: The Omega Singularity Engine — backend (Flask).
Five-model ensemble: A Monte Carlo GBM, B BSM Greeks + IV rank, C order-flow proxy,
D technical confluence, E NLP catalyst. Educational tool, not financial advice."""
import os, time, math, threading, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np, pandas as pd, pytz, requests, yfinance as yf
from scipy.stats import norm
from flask import Flask, jsonify, render_template

import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("omega")
app = Flask(__name__)

WATCHLIST = os.environ.get("WATCHLIST", "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMD,AMZN,META,GOOGL,NFLX,AVGO,PLTR,COIN,JPM").split(",")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
RF = float(os.environ.get("RISK_FREE", "0.045"))
N_PATHS, REFRESH_SEC = 10_000, 15
WEIGHTS = {"A": 0.30, "B": 0.15, "C": 0.20, "D": 0.25, "E": 0.10}
TARGET_DELTA = 0.35
NY = pytz.timezone("America/New_York")
sia = SentimentIntensityAnalyzer()
rng = np.random.default_rng()

# ---------------- TTL cache ----------------
_cache, _lock = {}, threading.Lock()
def cached(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val

# ---------------- Model B: Black-Scholes-Merton ----------------
def bsm(S, K, T, sig, kind, r=RF):
    T, sig = max(T, 1e-6), max(sig, 1e-4)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    pdf = norm.pdf(d1)
    if kind == "CALL":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * pdf * sig / (2 * sq) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sig / (2 * sq) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return dict(price=price, delta=delta, gamma=pdf / (S * sig * sq),
                theta=theta / 365, vega=S * pdf * sq / 100)

# ---------------- Model A: Monte Carlo GBM ----------------
def monte_carlo(S0, K, T, sigma, mu, kind):
    steps = max(int(T * 365 * 6.5), 8)
    dt = T / steps
    z = rng.standard_normal((N_PATHS, steps))
    logp = np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z, axis=1)
    ST = S0 * np.exp(logp[:, -1])
    itm = (ST > K) if kind == "CALL" else (ST < K)
    path_max, path_min = S0 * np.exp(logp.max(axis=1)), S0 * np.exp(logp.min(axis=1))
    touch = (path_max >= K) if kind == "CALL" else (path_min <= K)
    return float(itm.mean() * 100), float(touch.mean() * 100)

# ---------------- Data ----------------
def get_daily(t):
    return cached(f"d:{t}", 300, lambda: yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True))
def get_intraday(t):
    return cached(f"i:{t}", 60, lambda: yf.Ticker(t).history(period="5d", interval="5m", auto_adjust=True))

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

# ---------------- Model C: institutional flow proxy (signed -1..1) ----------------
def model_c(daily, intra):
    c, v = daily["Close"], daily["Volume"]
    rvol = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if v.iloc[-21:-1].mean() else 1.0
    day = intra[intra.index.date == intra.index[-1].date()]
    tp = (day["High"] + day["Low"] + day["Close"]) / 3
    vwap = float((tp * day["Volume"]).sum() / max(day["Volume"].sum(), 1))
    price = float(day["Close"].iloc[-1])
    vwap_sig = float(np.clip((price / vwap - 1) * 100, -1, 1))
    ret = float(c.iloc[-1] / c.iloc[-2] - 1)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    p_slope, o_slope = c.iloc[-1] - c.iloc[-11], obv.iloc[-1] - obv.iloc[-11]
    if p_slope > 0 and o_slope < 0: div = -1.0      # distribution
    elif p_slope < 0 and o_slope > 0: div = 1.0     # accumulation
    else: div = 0.5 * np.sign(o_slope)
    sig = 0.4 * vwap_sig + 0.3 * np.sign(ret) * min(rvol / 2, 1) + 0.3 * div
    return float(np.clip(sig, -1, 1)), dict(rvol=rvol, vwap=vwap, div=int(div), price=price)

# ---------------- Model D: technical confluence (signed -1..1) ----------------
def model_d(daily):
    c = daily["Close"]
    stack = lambda s: 1.0 if (ema(s, 9).iloc[-1] > ema(s, 21).iloc[-1] > ema(s, 50).iloc[-1]) else \
        (-1.0 if (ema(s, 9).iloc[-1] < ema(s, 21).iloc[-1] < ema(s, 50).iloc[-1]) else 0.0)
    wk = c.resample("W").last().dropna()
    r = rsi(c); r_now = float(r.iloc[-1])
    w = c.iloc[-12:]; rw = r.iloc[-12:]
    div = 0.0
    if w.iloc[-1] <= w.min() * 1.005 and rw.iloc[-1] > rw.min() + 2: div = 1.0
    if w.iloc[-1] >= w.max() * 0.995 and rw.iloc[-1] < rw.max() - 2: div = -1.0
    macd = ema(c, 12) - ema(c, 26); hist = macd - ema(macd, 9)
    m_sig = 0.5 * np.sign(hist.iloc[-1]) + 0.5 * np.sign(hist.iloc[-1] - hist.iloc[-2])
    rsi_sig = float(np.clip((r_now - 50) / 25, -1, 1)) * 0.6 + 0.4 * div
    sig = 0.30 * stack(c) + 0.20 * (stack(wk) if len(wk) > 50 else (1.0 if ema(wk, 9).iloc[-1] > ema(wk, 21).iloc[-1] else -1.0)) \
        + 0.25 * rsi_sig + 0.25 * m_sig
    return float(np.clip(sig, -1, 1)), dict(rsi=r_now, macd_hist=float(hist.iloc[-1]), rsi_div=int(div))

# ---------------- Model E: NLP catalyst (signed -1..1) ----------------
def model_e(t):
    if not NEWSAPI_KEY:
        return 0.0, dict(n=0, note="ไม่ได้ตั้งค่า NEWSAPI_KEY")
    def fetch():
        r = requests.get("https://newsapi.org/v2/everything", timeout=8, params=dict(
            q=f'"{t}" stock', language="en", pageSize=20, sortBy="publishedAt", apiKey=NEWSAPI_KEY))
        return r.json().get("articles", []) if r.ok else []
    try:
        arts = cached(f"n:{t}", 900, fetch)
    except Exception as ex:
        log.warning("news %s failed: %s", t, ex)
        return 0.0, dict(n=0, note="ดึงข่าวไม่สำเร็จ")
    sc = [sia.polarity_scores(f"{a.get('title') or ''}. {a.get('description') or ''}")["compound"] for a in arts]
    sc = [s for s in sc if abs(s) > 0.05]
    return (float(np.clip(np.mean(sc) * 1.5, -1, 1)) if sc else 0.0), dict(n=len(arts), note="")

# ---------------- Option selection ----------------
def pick_expiry(tk):
    now = datetime.now(NY)
    best = None
    for e in tk.options:
        exp = NY.localize(datetime.strptime(e, "%Y-%m-%d").replace(hour=16))
        dte = (exp - now).total_seconds() / 86400
        if 2 <= dte <= 10:
            return e, dte
        if dte >= 1 and best is None: best = (e, dte)
    return best or (None, None)

def analyze(t):
    tk = yf.Ticker(t)
    daily, intra = get_daily(t), get_intraday(t)
    if daily.empty or len(daily) < 60 or intra.empty:
        return None
    spot = float(intra["Close"].iloc[-1])
    exp, dte = cached(f"x:{t}", 600, lambda: pick_expiry(tk))
    if not exp: return None
    chain = cached(f"c:{t}:{exp}", 60, lambda: tk.option_chain(exp))
    T = dte / 365

    c_sig, c_info = model_c(daily, intra)
    d_sig, d_info = model_d(daily)
    e_sig, e_info = model_e(t)
    kind = "CALL" if (c_sig + d_sig + e_sig) >= 0 else "PUT"
    sgn = 1 if kind == "CALL" else -1

    df = (chain.calls if kind == "CALL" else chain.puts).copy()
    df = df[(df["impliedVolatility"] > 0.01) & ((df["ask"] > 0) | (df["lastPrice"] > 0))]
    df = df[df["strike"] > spot] if kind == "CALL" else df[df["strike"] < spot]
    if df.empty: return None
    atm = pd.concat([chain.calls, chain.puts]); atm = atm[atm["impliedVolatility"] > 0.01]
    atm = atm.iloc[(atm["strike"] - spot).abs().argsort()[:4]]
    atm_iv = float(atm["impliedVolatility"].mean())
    em = max(spot * atm_iv * math.sqrt(T), spot * 0.002)   # 1-sigma expected move
    target = spot + sgn * 0.5 * em

    df["delta"] = [abs(bsm(spot, r.strike, T, r.impliedVolatility, kind)["delta"]) for r in df.itertuples()]
    band = df[(df["delta"] >= 0.30) & (df["delta"] <= 0.40)]
    pool = band if not band.empty else df
    pool = pool.assign(dist=(pool["strike"] - target).abs() / em + (pool["delta"] - TARGET_DELTA).abs() * 4)
    row = pool.sort_values("dist").iloc[0]
    K, iv = float(row["strike"]), float(row["impliedVolatility"])
    mid = float((row["bid"] + row["ask"]) / 2) if row["bid"] > 0 and row["ask"] > 0 else float(row["lastPrice"] or row["ask"])
    g = bsm(spot, K, T, iv, kind)

    # IV rank / percentile against 6-month rolling 30d realised-vol distribution (proxy)
    lr = np.log(daily["Close"]).diff()
    hv = (lr.rolling(30).std() * math.sqrt(252)).dropna().iloc[-126:]
    ivr = float(np.clip((atm_iv - hv.min()) / max(hv.max() - hv.min(), 1e-9), 0, 1) * 100)
    ivp = float((hv < atm_iv).mean() * 100)
    hv20 = float(lr.iloc[-20:].std() * math.sqrt(252))

    # Model A
    sigma = 0.7 * iv + 0.3 * hv20
    mu = RF + 0.2 * float(np.clip((c_sig + d_sig + e_sig) / 3, -1, 1))
    win_a, touch = monte_carlo(spot, K, T, sigma, mu, kind)

    # Model scores 0-100 (probability-style support for the chosen direction)
    support = lambda s: float(np.clip(50 + 50 * sgn * s, 0, 100))
    decay = abs(g["theta"]) / max(mid, 0.01)
    score_b = float(np.clip(0.6 * (100 - ivr) + 0.4 * np.clip(100 - decay * 400, 0, 100), 0, 100))
    scores = {"A": win_a, "B": score_b, "C": support(c_sig), "D": support(d_sig), "E": support(e_sig)}
    cons = sum(WEIGHTS[k] * v for k, v in scores.items())
    verdict = "ULTIMATE" if cons >= 85 else ("HIGH" if cons >= 70 else "NONE")

    return dict(
        ticker=t, price=round(spot, 2), type=kind, strike=K, expiry=exp, dte=round(dte, 1),
        premium=round(mid, 2), win_rate=round(cons, 1), verdict=verdict, scores={k: round(v, 1) for k, v in scores.items()},
        mc_itm=round(win_a, 1), mc_touch=round(touch, 1),
        expected_move=round(em, 2), em_low=round(spot - em, 2), em_high=round(spot + em, 2),
        greeks={k: round(v, 4) for k, v in g.items() if k != "price"}, iv=round(iv * 100, 1), atm_iv=round(atm_iv * 100, 1),
        iv_rank=round(ivr, 1), iv_pct=round(ivp, 1), iv_tag="ถูก" if ivr < 30 else ("แพง" if ivr > 70 else "กลาง"),
        flow=c_info, tech=d_info, news=e_info, catalyst=round(50 + 50 * e_sig, 1),
        breakeven=round(K + sgn * mid, 2),
    )

# ---------------- Scanner loop ----------------
STATE = {"items": [], "updated": None, "cycle_sec": 0, "errors": []}
def scan_once():
    t0, items, errs = time.time(), [], []
    def work(t):
        try: return analyze(t.strip().upper())
        except Exception as ex:
            errs.append(f"{t}: {ex}"); log.warning("scan %s failed: %s", t, ex); return None
    with ThreadPoolExecutor(max_workers=6) as ex:
        items = [r for r in ex.map(work, WATCHLIST) if r]
    items.sort(key=lambda r: r["win_rate"], reverse=True)
    STATE.update(items=items, updated=datetime.now(pytz.utc).isoformat(), cycle_sec=round(time.time() - t0, 1), errors=errs[:5])

def loop():
    while True:
        s = time.time()
        try: scan_once()
        except Exception as ex: log.exception(ex)
        time.sleep(max(REFRESH_SEC - (time.time() - s), 1))

_started = False
def start():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=loop, daemon=True).start()
start()

def clean(o):
    """Make state JSON-safe: NaN/inf -> null, numpy scalars -> Python."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o); return f if math.isfinite(f) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

@app.route("/")
def index(): return render_template("index.html", refresh=REFRESH_SEC)
@app.route("/api/scan")
def api_scan(): return jsonify(clean(STATE))
@app.route("/health")
def health(): return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
