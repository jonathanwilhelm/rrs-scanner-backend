from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import numpy as np
import asyncio
import logging
import time

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
    atr_avg    = ((my_atr + bench_atr) / 2) * mult
    if atr_avg <= 0:
        atr_avg = my_atr or 1.0
    my_c   = my_bars[:, 2]
    bn_c   = bench_bars[:, 2]
    series = ((my_c[length:] - my_c[:-length]) - (bn_c[length:] - bn_c[:-length])) / atr_avg
    alpha  = 2 / (3 + 1)
    ema    = float(series[0])
    for v in series[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return float(np.clip(ema, -20, 20))


def download_with_retry(symbol: str, interval: str, period: str, retries: int = 3):
    """Télécharge les données avec retry et délai croissant."""
    for attempt in range(retries):
        try:
            df = yf.download(
                symbol,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=True,
                timeout=30,
            )
            if df is not None and not df.empty and len(df) >= 10:
                return df
            logger.warning(f"{symbol} tentative {attempt+1}: données vides")
        except Exception as e:
            logger.warning(f"{symbol} tentative {attempt+1} erreur: {e}")
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
    return None


async def fetch_symbol(symbol: str, interval: str, period: str):
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: download_with_retry(symbol, interval, period)
        )
        if df is None:
            return symbol, None, None, None

        # Aplatir les colonnes MultiIndex si présentes (yfinance >= 0.2.40)
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        needed = [c for c in ["High", "Low", "Close"] if c in df.columns]
        if len(needed) < 3:
            return symbol, None, None, None

        bars  = df[["High", "Low", "Close"]].dropna().to_numpy(dtype=float)
        if len(bars) < 10:
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


@app.get("/scan")
async def scan(
    symbols:  str   = Query(...),
    bench:    str   = Query("SPY"),
    length:   int   = Query(14, ge=5, le=50),
    mult:     float = Query(1.0, ge=0.5, le=3.0),
    interval: str   = Query("5m"),
    period:   str   = Query("5d"),
):
    sym_list  = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    bench_sym = bench.strip().upper()

    if len(sym_list) > 750:
        return JSONResponse({"error": "Maximum 750 symboles"}, status_code=400)

    # Télécharger le benchmark en premier, séparément
    logger.info(f"Téléchargement benchmark {bench_sym} interval={interval} period={period}")
    _, bench_bars, _, _ = await fetch_symbol(bench_sym, interval, period)

    if bench_bars is None:
        # Fallback : essayer avec period plus long
        logger.warning(f"Retry benchmark {bench_sym} avec period=1mo")
        _, bench_bars, _, _ = await fetch_symbol(bench_sym, interval, "1mo")

    if bench_bars is None:
        return JSONResponse(
            {"error": f"Impossible de récupérer le benchmark {bench_sym}. "
                      f"Vérifiez que le marché est ouvert ou essayez interval=1d period=1mo"},
            status_code=502
        )

    logger.info(f"Benchmark OK: {len(bench_bars)} barres. Scan de {len(sym_list)} symboles...")

    # Téléchargement parallèle de tous les symboles
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
        "benchmark": bench_sym,
        "length":    length,
        "mult":      mult,
        "interval":  interval,
        "period":    period,
        "count":     len(output),
        "results":   output,
    }
