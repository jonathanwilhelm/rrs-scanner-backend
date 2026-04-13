from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import numpy as np
import asyncio
import logging

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
    atr = np.mean(tr[:n])
    for i in range(n, len(tr)):
        atr = (atr * (length - 1) + tr[i]) / length
    return float(atr) if atr > 0 else 1.0


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
    ema    = series[0]
    for v in series[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(np.clip(ema, -20, 20))


async def fetch_symbol(symbol, interval, period):
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(symbol, interval=interval, period=period,
                                 progress=False, auto_adjust=True)
        )
        if df.empty or len(df) < 20:
            return symbol, None, None, None
        bars  = df[["High", "Low", "Close"]].dropna().to_numpy()
        price = float(bars[-1, 2])
        opens = df["Open"].dropna().to_numpy()
        chg   = round((price - float(opens[-1])) / float(opens[-1]) * 100, 2) if len(opens) else 0.0
        return symbol, bars, price, chg
    except Exception as e:
        logger.warning(f"Erreur {symbol}: {e}")
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
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(sym_list) > 750:
        return JSONResponse({"error": "Maximum 750 symboles"}, status_code=400)

    all_syms = list(set([bench.upper()] + sym_list))
    tasks    = [fetch_symbol(s, interval, period) for s in all_syms]
    results  = await asyncio.gather(*tasks)

    data_map, price_map, chg_map = {}, {}, {}
    for sym, bars, price, chg in results:
        if bars is not None:
            data_map[sym]  = bars
            price_map[sym] = price
            chg_map[sym]   = chg

    bench_sym = bench.upper()
    if bench_sym not in data_map:
        return JSONResponse({"error": f"Benchmark {bench_sym} indisponible"}, status_code=502)

    bench_bars = data_map[bench_sym]
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
            "rrs":   round(rrs, 2) if rrs is not None else None,
            "price": round(price_map[sym], 2),
            "chg":   chg_map[sym],
            "error": None,
        })

    output.sort(key=lambda x: (x["rrs"] is None, -(x["rrs"] or -999)))
    return {"benchmark": bench_sym, "length": length, "mult": mult,
            "interval": interval, "count": len(output), "results": output}
