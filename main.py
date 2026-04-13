from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import asyncio
import logging
import urllib.request
import csv
import io
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def wilder_atr(bars, length):
    highs, lows, closes = bars[:, 0], bars[:, 1], bars[:, 2]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
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
    atr_avg = ((wilder_atr(my_bars, length) + wilder_atr(bench_bars, length)) / 2) * mult or 1.0
    my_c = my_bars[:, 2]
    bn_c = bench_bars[:, 2]
    series = ((my_c[length:] - my_c[:-length]) - (bn_c[length:] - bn_c[:-length])) / atr_avg
    ema = float(series[0])
    for v in series[1:]:
        ema = 0.5 * float(v) + 0.5 * ema
    return float(np.clip(ema, -20, 20))


def get_bars(symbol: str, days: int = 120):
    """
    Télécharge les données OHLC via l'API CSV de Stooq.
    Utilise uniquement urllib — aucune dépendance externe.
    """
    end   = datetime.today()
    start = end - timedelta(days=days)
    d1 = start.strftime("%Y%m%d")
    d2 = end.strftime("%Y%m%d")

    # Stooq: suffixe .US pour les actions américaines
    sym = f"{symbol}.US".lower()
    url = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")

        if "No data" in raw or len(raw) < 50:
            logger.warning(f"{symbol}: pas de données Stooq")
            return None, None, None

        reader = csv.DictReader(io.StringIO(raw))
        rows   = list(reader)
        if not rows or len(rows) < 15:
            return None, None, None

        # Stooq retourne en ordre décroissant — on inverse
        rows.reverse()

        highs, lows, closes, opens = [], [], [], []
        for r in rows:
            try:
                highs.append(float(r["High"]))
                lows.append(float(r["Low"]))
                closes.append(float(r["Close"]))
                opens.append(float(r.get("Open", r["Close"])))
            except (ValueError, KeyError):
                continue

        if len(closes) < 15:
            return None, None, None

        bars  = np.column_stack([highs, lows, closes])
        price = closes[-1]
        chg   = round((closes[-1] - opens[-1]) / opens[-1] * 100, 2) if opens[-1] > 0 else 0.0
        logger.info(f"{symbol} OK: {len(bars)} barres, prix={price:.2f}")
        return bars, price, chg

    except Exception as e:
        logger.error(f"{symbol} erreur: {e}")
        return None, None, None


@app.get("/")
def root():
    return {"status": "ok", "name": "Trinity RRS Scanner", "source": "Stooq CSV"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "Trinity RRS Backend opérationnel"}

@app.get("/test/{symbol}")
async def test(symbol: str):
    loop = asyncio.get_event_loop()
    bars, price, chg = await loop.run_in_executor(None, lambda: get_bars(symbol.upper()))
    if bars is None:
        return {"symbol": symbol.upper(), "status": "ERREUR", "bars": 0}
    return {"symbol": symbol.upper(), "status": "OK", "bars": len(bars),
            "price": round(price, 2), "chg": chg}

@app.get("/scan")
async def scan(
    symbols: str   = Query(...),
    bench:   str   = Query("SPY"),
    length:  int   = Query(14, ge=5, le=50),
    mult:    float = Query(1.0, ge=0.5, le=3.0),
    days:    int   = Query(120, ge=30, le=365),
):
    sym_list  = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    bench_sym = bench.strip().upper()

    if len(sym_list) > 750:
        return JSONResponse({"error": "Maximum 750 symboles"}, status_code=400)

    loop = asyncio.get_event_loop()

    bench_bars, _, _ = await loop.run_in_executor(None, lambda: get_bars(bench_sym))
    if bench_bars is None:
        return JSONResponse({"error": f"Benchmark {bench_sym} indisponible"}, status_code=502)

    async def fetch(sym):
        b, p, c = await loop.run_in_executor(None, lambda: get_bars(sym))
        return sym, b, p, c

    results = await asyncio.gather(*[fetch(s) for s in sym_list])

    output = []
    for sym, bars, price, chg in results:
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
    return {"benchmark": bench_sym, "source": "Stooq",
            "count": len(output), "results": output}
