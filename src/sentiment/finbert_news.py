import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "yiyanghkust/finbert-tone"

# Project root (works on Render and locally)
_ROOT = Path(__file__).resolve().parent.parent.parent


def score_news(ticker="MSFT", batch_size=32, device=None):
    """
    Score news sentiment using FinBERT with batched processing

    Args:
        ticker: Stock ticker symbol
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu', auto-detects if None
    """
    DATA_PATH = _ROOT / "data" / "news"
    OUT_PATH = _ROOT / "data" / "processed" / "news"

    # Load data
    input_path = DATA_PATH / f"{ticker}.csv"
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Validate required columns
    if "title" not in df.columns:
        logger.error("'title' column not found in data")
        return

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Prepare texts
    texts = df["title"].fillna("").tolist()

    # Process in batches
    all_sentiments = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Scoring {ticker}"):
        batch_texts = texts[i:i + batch_size]

        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Move to CPU and convert to list
            all_sentiments.extend(probs.cpu().tolist())

        except Exception as e:
            logger.warning(f"Error in batch {i}: {e}")
            # Add neutral sentiment as fallback
            all_sentiments.extend([[0, 1.0, 0]] * len(batch_texts))

    # Create sentiment DataFrame
    # FinBERT yiyanghkust/finbert-tone output order: index 0=neutral, 1=positive, 2=negative
    sent_df = pd.DataFrame(all_sentiments, columns=["neu", "pos", "neg"])

    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

    # Save results
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUT_PATH / f"{ticker}.csv"
    result_df.to_csv(output_path, index=False)

    logger.info(f"[✓] Scored {len(df)} news articles for {ticker}")
    logger.info(f"    Saved to: {output_path}")

    # Memory cleanup
    del model
    torch.cuda.empty_cache()

    return result_df


def score_multiple_tickers(tickers, **kwargs):
    """Score multiple tickers"""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = score_news(ticker, **kwargs)
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
    return results


if __name__ == "__main__":
    # Example usage
    score_news("AAPL", batch_size=16)

    # Or multiple tickers
    # tickers = ["AAPL", "MSFT", "GOOGL"]
    # score_multiple_tickers(tickers, batch_size=32)