from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import numpy as np
import asyncio
import logging
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trinity RRS Scanner API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session avec User-Agent navigateur pour contourner le blocage Yahoo
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
})


def wilder_atr(bars: np.ndarray, length: int) -> float:
    highs, lows, closes = bars[:, 0], bars[:, 1], bars[:, 2]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    n = min(length, len(tr))
    atr = float(np.mean(tr[:n]))
    for i in range(n, len(tr)):
        atr = (atr * (length - 1) + float(tr[i])) / length
    return atr if atr > 0 else 1.0


def calc_rrs(my_bars, bench_bars, length, mult):
    n = min(len(my_bars), len(bench_bars))
    if n < length + 5:
        return None
    my_bars    = my_bars[-n:]
    bench_bars = bench_bars[-n:]
    my_atr     = wilder_atr(my_bars, length)
    bench_atr  = wilder_atr(bench_bars, length)
    atr_avg    = ((my_atr + bench_atr) / 2) * mult or my_atr or 1.0
    my_c   = my_bars[:, 2]
    bn_c   = bench_bars[:, 2]
    series = ((my_c[length:] - my_c[:-length]) - (bn_c[length:] - bn_c[:-length])) / atr_avg
    alpha  = 2 / (3 + 1)
    ema    = float(series[0])
    for v in series[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return float(np.clip(ema, -20, 20))


def download_symbol(symbol: str, interval: str, period: str):
    """Télécharge avec session custom et retry x3."""
    combos = [(interval, period), ("1d", "1mo"), ("1d", "3mo")]
    for iv, pr in combos:
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol, session=SESSION)
                df = ticker.history(interval=iv, period=pr, auto_adjust=True)
                if df is not None and not df.empty and len(df) >= 10:
                    logger.info(f"{symbol} OK: {len(df)} barres ({iv}/{pr})")
                    return df
                logger.warning(f"{symbol} vide tentative {attempt+1} ({iv}/{pr})")
            except Exception as e:
                logger.warning(f"{symbol} erreur tentative {attempt+1}: {e}")
            time.sleep(1)
    return None


def df_to_bars(df):
    """Convertit un DataFrame yfinance en array numpy HLC."""
    cols = {c.lower(): c for c in df.columns}
    h = cols.get("high", "High")
    l = cols.get("low", "Low")
    c = cols.get("close", "Close")
    bars = df[[h, l, c]].dropna().to_numpy(dtype=float)
    return bars if len(bars) >= 10 else None


async def fetch_symbol(symbol: str, interval: str, period: str):
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: download_symbol(symbol, interval, period))
        if df is None:
            return symbol, None, None, None
        bars  = df_to_bars(df)
        if bars is None:
            return symbol, None, None, None
        price = float(bars[-1, 2])
        chg   = 0.0
        if "Open" in df.columns:
            opens = df["Open"].dropna().to_numpy(dtype=float)
            if len(opens) > 0 and opens[-1] > 0:
                chg = round((price - float(opens[-1])) / float(opens[-1]) * 100, 2)
        return symbol, bars, price, chg
    except Exception as e:
        logger.error(f"fetch_symbol {symbol}: {e}")
        return symbol, None, None, None


@app.get("/health")
def health():
    return {"status": "ok", "message": "Trinity RRS Backend opérationnel"}

@app.get("/")
def root():
    return {"name": "Trinity RRS Scanner API", "version": "1.0.0"}

@app.get("/test/{symbol}")
async def test_symbol(symbol: str):
    """Debug: teste le téléchargement d'un symbole."""
    sym, bars, price, chg = await fetch_symbol(symbol.upper(), "1d", "1mo")
    if bars is None:
        return {"symbol": sym, "status": "ERREUR", "bars": 0}
    return {"symbol": sym, "status": "OK", "bars": len(bars), "last_price": round(price, 2)}

@app.get("/scan")
async def scan(
    symbols:  str   = Query(...),
    bench:    str   = Query("SPY"),
    length:   int   = Query(14, ge=5, le=50),
    mult:     float = Query(1.0, ge=0.5, le=3.0),
    interval: str   = Query("1d"),
    period:   str   = Query("1mo"),
):
    sym_list  = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    bench_sym = bench.strip().upper()

    if len(sym_list) > 750:
        return JSONResponse({"error": "Maximum 750 symboles"}, status_code=400)

    logger.info(f"Scan: bench={bench_sym} interval={interval} period={period} symbols={len(sym_list)}")

    _, bench_bars, _, _ = await fetch_symbol(bench_sym, interval, period)
    if bench_bars is None:
        return JSONResponse(
            {"error": f"Benchmark {bench_sym} indisponible — essayez /test/{bench_sym} pour diagnostiquer"},
            status_code=502
        )

    tasks   = [fetch_symbol(s, interval, period) for s in sym_list]
    results = await asyncio.gather(*tasks)

    output = []
    for sym, bars, price, chg in results:
        if sym == bench_sym:
            continue
        if bars is None:
            output.append({"symbol": sym, "rrs": None, "price": None, "chg": None, "error": "no_data"})
            continue
        rrs = calc_rrs(bars, bench_bars, length, mult)
        output.append({
            "symbol": sym,
            "rrs":    round(rrs, 2) if rrs is not None else None,
            "price":  round(price, 2),
            "chg":    chg,
            "error":  None,
        })

    output.sort(key=lambda x: (x["rrs"] is None, -(x["rrs"] or -999)))
    return {
        "benchmark": bench_sym, "length": length, "mult": mult,
        "interval": interval, "period": period,
        "count": len(output), "results": output,
    }
