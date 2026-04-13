"""
Trinity RRS Scanner — Backend Python
Algorithme identique au Pine Script "Real Relative Strength Indicator"
Déployez sur Render.com (gratuit) en 5 minutes.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import numpy as np
import asyncio
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trinity RRS Scanner API", version="1.0.0")

# CORS ouvert pour que le scanner dans le navigateur puisse appeler cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def wilder_atr(bars: np.ndarray, length: int) -> float:
    """ATR de Wilder — même calcul que Pine Script ta.atr()"""
    highs, lows, closes = bars[:, 0], bars[:, 1], bars[:, 2]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    n = min(length, len(tr))
    atr = np.mean(tr[:n])
    for i in range(n, len(tr)):
        atr = (atr * (length - 1) + tr[i]) / length
    return float(atr) if atr > 0 else 1.0


def calc_rrs(my_bars: np.ndarray, bench_bars: np.ndarray,
             length: int, mult: float) -> float | None:
    """
    Calcul exact du RRS Pine Script :
    rrs = (myMom - benchMom) / atrAvg
    rrsMA = EMA(rrs, 3)  — clampé à [-20, 20]
    """
    n = min(len(my_bars), len(bench_bars))
    if n < length + 5:
        return None

    my_bars  = my_bars[-n:]
    bench_bars = bench_bars[-n:]

    my_atr    = wilder_atr(my_bars,    length)
    bench_atr = wilder_atr(bench_bars, length)
    atr_avg   = ((my_atr + bench_atr) / 2) * mult
    if atr_avg <= 0:
        atr_avg = my_atr or 1.0

    my_closes    = my_bars[:, 2]
    bench_closes = bench_bars[:, 2]

    # Série RRS brute
    rrs_series = (
        (my_closes[length:] - my_closes[:-length]) -
        (bench_closes[length:] - bench_closes[:-length])
    ) / atr_avg

    # EMA(3) — alpha = 2/(3+1) = 0.5
    alpha = 2 / (3 + 1)
    ema = rrs_series[0]
    for v in rrs_series[1:]:
        ema = alpha * v + (1 - alpha) * ema

    return float(np.clip(ema, -20, 20))


async def fetch_symbol(symbol: str, interval: str, period: str) -> tuple[str, np.ndarray | None, float | None, float | None]:
    """Télécharge les OHLC via yfinance de façon asynchrone"""
    try:
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(
            None,
            lambda: yf.download(symbol, interval=interval, period=period,
                                 progress=False, auto_adjust=True)
        )
        if ticker.empty or len(ticker) < 20:
            return symbol, None, None, None

        bars = ticker[["High", "Low", "Close"]].dropna().to_numpy()
        price = float(bars[-1, 2])

        # % variation du jour
        open_prices = ticker["Open"].dropna().to_numpy()
        day_open = float(open_prices[-1]) if len(open_prices) > 0 else price
        chg = round((price - day_open) / day_open * 100, 2)

        return symbol, bars, price, chg

    except Exception as e:
        logger.warning(f"Erreur {symbol}: {e}")
        return symbol, None, None, None


@app.get("/scan")
async def scan(
    symbols: str = Query(..., description="Symboles séparés par virgule, ex: AAPL,MSFT,NVDA"),
    bench:   str = Query("SPY",  description="Symbole benchmark"),
    length:  int = Query(14,     description="Période momentum (5–50)", ge=5, le=50),
    mult:   float = Query(1.0,   description="Multiplicateur ATR (0.5–3.0)", ge=0.5, le=3.0),
    interval:str = Query("5m",   description="Intervalle: 1m 5m 15m 1h 1d"),
    period:  str = Query("5d",   description="Historique: 1d 5d 1mo 3mo"),
):
    """
    Endpoint principal — retourne le score RRS pour chaque symbole.

    Exemple d'appel :
    GET /scan?symbols=AAPL,MSFT,NVDA,TSLA&bench=SPY&length=14&mult=1.0&interval=5m&period=5d
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if len(sym_list) > 750:
        return JSONResponse({"error": "Maximum 750 symboles par requête"}, status_code=400)

    # Téléchargement concurrent (benchmark + tous les symboles en parallèle)
    all_symbols = list(set([bench.upper()] + sym_list))
    tasks = [fetch_symbol(s, interval, period) for s in all_symbols]
    results_raw = await asyncio.gather(*tasks)

    # Organiser les résultats
    data_map: dict[str, np.ndarray] = {}
    price_map: dict[str, float]     = {}
    chg_map:   dict[str, float]     = {}

    for sym, bars, price, chg in results_raw:
        if bars is not None:
            data_map[sym] = bars
            price_map[sym] = price
            chg_map[sym]   = chg

    bench_sym = bench.upper()
    if bench_sym not in data_map:
        return JSONResponse({"error": f"Impossible de récupérer le benchmark {bench_sym}"}, status_code=502)

    bench_bars = data_map[bench_sym]

    # Calcul RRS pour chaque symbole
    output = []
    for sym in sym_list:
        if sym == bench_sym:
            continue
        if sym not in data_map:
            output.append({"symbol": sym, "rrs": None, "price": None, "chg": None, "error": "no_data"})
            continue

        rrs = calc_rrs(data_map[sym], bench_bars, length, mult)
        output.append({
            "symbol": sym,
            "rrs":    round(rrs, 2) if rrs is not None else None,
            "price":  round(price_map[sym], 2),
            "chg":    chg_map[sym],
            "error":  None,
        })

    # Trier par RRS décroissant
    output.sort(key=lambda x: (x["rrs"] is None, -(x["rrs"] or -999)))

    return {
        "benchmark": bench_sym,
        "length":    length,
        "mult":      mult,
        "interval":  interval,
        "count":     len(output),
        "results":   output,
    }


@app.get("/health")
def health():
    return {"status": "ok", "message": "Trinity RRS Backend opérationnel"}


@app.get("/")
def root():
    return {
        "name":    "Trinity RRS Scanner API",
        "version": "1.0.0",
        "endpoints": {
            "/scan":   "Calcul RRS (GET)",
            "/health": "Statut du serveur (GET)",
            "/docs":   "Documentation interactive (GET)",
        }
    }
