"""
Tester: run full pipeline for top stocks + cryptos and save all final analyses to one CSV.

Usage (from project root):
    python run_full_pipeline_tester.py

Output: data/full_analysis_results.csv (or path set in OUTPUT_CSV).
Partial results are saved after each ticker so a crash doesn't lose progress.
"""

import sys
import traceback
import pandas as pd
from pathlib import Path

# Ticker lists
TOP_20_STOCKS = [

    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "NVDA",   # NVIDIA Corporation
    "GOOGL",  # Alphabet Inc. (Class A)
    "AMZN",   # Amazon.com Inc.
    "META",   # Meta Platforms Inc.
    "TSLA",   # Tesla Inc.
    "BRK-B",  # Berkshire Hathaway Inc. (Class B)
    "LLY",    # Eli Lilly and Company
    "V",      # Visa Inc.
    "JPM",    # JPMorgan Chase & Co.
    "XOM",    # Exxon Mobil Corporation
    "UNH",    # UnitedHealth Group Inc.
    "WMT",    # Walmart Inc.
    "JNJ",    # Johnson & Johnson
    "AVGO",   # Broadcom Inc.
    "MA",     # Mastercard Incorporated
    "PG",     # Procter & Gamble Company
    "HD",     # Home Depot Inc.
    "ORCL",   # Oracle Corporation
]

TOP_10_CRYPTOS = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "BNB-USD",   # Binance Coin
    "SOL-USD",   # Solana
    "XRP-USD",   # Ripple
    "DOGE-USD",  # Dogecoin
    "ADA-USD",   # Cardano
    "AVAX-USD",  # Avalanche
    "DOT-USD",   # Polkadot
    "SHIB-USD",  # Shiba Inu
]

OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "full_analysis_results.csv"


def _flatten_result(ticker: str, category: str, result: dict) -> dict:
    """Turn full_analysis result into a single flat row for CSV."""
    reasons = result.get("reasons")
    risks = result.get("risks")
    row = {
        "ticker": ticker,
        "category": category,
        "recommendation": result.get("recommendation", ""),
        "signal": result.get("signal", ""),
        "confidence": result.get("confidence", ""),
        "prob_up": result.get("prob_up"),
        "date": result.get("date", ""),
        "analysis": result.get("analysis", ""),
        "pred_return": result.get("pred_return"),
        "reasons": " | ".join(reasons) if isinstance(reasons, list) else str(reasons) if reasons else "",
        "risks": " | ".join(risks) if isinstance(risks, list) else str(risks) if risks else "",
        "error": "",
    }
    return row


def run_one(ticker: str):
    """Run full pipeline + analysis for one ticker. Returns (result_dict, None) or (None, error_message)."""
    from fastapi import HTTPException
    from src.api.main import _run_full_analysis
    try:
        return _run_full_analysis(ticker.strip().upper()), None
    except HTTPException as e:
        msg = getattr(e, "detail", str(e))
        print(f"   ❌ {ticker}: {msg}", flush=True)
        return None, msg
    except Exception as e:
        print(f"   ❌ {ticker}: {e}", flush=True)
        traceback.print_exc()
        return None, str(e)


def main():
    all_tickers = [
        *[(t, "stock") for t in TOP_20_STOCKS],
        *[(t, "crypto") for t in TOP_10_CRYPTOS],
    ]
    total = len(all_tickers)
    rows = []

    print(f"Running full pipeline + analysis for {total} tickers ({len(TOP_20_STOCKS)} stocks, {len(TOP_10_CRYPTOS)} cryptos).", flush=True)
    print("This may take a long time (price fetch, Reddit, news, sentiment, build_dataset, predict per ticker).\n", flush=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    for i, (ticker, category) in enumerate(all_tickers, 1):
        print(f"[{i}/{total}] {ticker} ({category})...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            result, error = run_one(ticker)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            print(f"   ❌ Unexpected error for {ticker}: {e}", flush=True)
            traceback.print_exc()
            error = str(e)
            result = None
        if result is not None:
            rows.append(_flatten_result(ticker, category, result))
            print(f"   ✅ signal={result.get('signal')} prob_up={result.get('prob_up')}", flush=True)
        else:
            rows.append({
                "ticker": ticker,
                "category": category,
                "recommendation": "",
                "signal": "",
                "confidence": "",
                "prob_up": None,
                "date": "",
                "analysis": "",
                "pred_return": None,
                "reasons": "",
                "risks": "",
                "error": error or f"Pipeline or analysis failed for {ticker}",
            })
        # Save after each ticker so a crash doesn't lose progress
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"   💾 Progress saved ({len(rows)}/{total})", flush=True)

    print(f"\n✅ Saved full analysis for {len(rows)} tickers to: {OUTPUT_CSV}", flush=True)
    return df


if __name__ == "__main__":
    main()
