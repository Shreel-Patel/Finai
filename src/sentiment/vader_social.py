import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nltk
import logging
from typing import Dict, List, Optional
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download VADER if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Consider financial-specific alternatives
# from transformers import pipeline
# financial_sentiment = pipeline("sentiment-analysis",
#                                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

class SocialSentimentAnalyzer:
    """Analyze sentiment from social media sources"""

    def __init__(self, use_financial_model: bool = False):
        """
        Args:
            use_financial_model: If True, use financial-tuned model instead of VADER
        """
        self.use_financial_model = use_financial_model

        if use_financial_model:
            self._setup_financial_model()
        else:
            self.sid = SentimentIntensityAnalyzer()

        # Map sources to their text columns
        self.SOURCES = {
            "reddit": "content",
            # "twitter": "text",
            # "stocktwits": "message",
            # Add more sources as needed
        }

    def _setup_financial_model(self):
        """Setup financial-specific sentiment model"""
        try:
            from transformers import pipeline
            logger.info("Loading financial sentiment model...")
            self.financial_model = pipeline(
                "sentiment-analysis",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                device=-1  # CPU; use 0 for GPU if available
            )
        except ImportError:
            logger.warning("Transformers not installed. Falling back to VADER.")
            self.use_financial_model = False
            self.sid = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if not text or pd.isna(text):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        if self.use_financial_model:
            try:
                result = self.financial_model(text[:512])[0]  # Truncate to model limit
                # Convert to VADER-like format
                label = result['label'].lower()
                score = result['score']
                return {
                    "neg": score if label == "negative" else 0.0,
                    "neu": score if label == "neutral" else 0.0,
                    "pos": score if label == "positive" else 0.0,
                    "compound": score if label == "positive" else -score
                }
            except Exception as e:
                logger.warning(f"Financial model failed: {e}. Falling back to VADER.")

        # Fallback to VADER
        return self.sid.polarity_scores(text)

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts efficiently"""
        # For VADER, we can't truly batch but can use list comprehension
        if not self.use_financial_model:
            return [self.analyze_text(text) for text in texts]
        else:
            # Financial model supports batch processing
            try:
                results = self.financial_model(texts, truncation=True)
                scores = []
                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    scores.append({
                        "neg": score if label == "negative" else 0.0,
                        "neu": score if label == "neutral" else 0.0,
                        "pos": score if label == "positive" else 0.0,
                        "compound": score if label == "positive" else -score
                    })
                return scores
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                return [self.analyze_text(text) for text in texts]

    def score_social(self, source: str, ticker: str,
                     data_dir: Optional[Path] = None,
                     output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Score sentiment for a specific source and ticker

        Args:
            source: 'reddit', 'twitter', etc.
            ticker: Stock ticker symbol
            data_dir: Directory containing raw data
            output_dir: Directory for processed data
        """
        # Setup paths
        _root = Path(__file__).resolve().parent.parent.parent
        if data_dir is None:
            data_dir = _root / "data" / source
        if output_dir is None:
            output_dir = _root / "data" / "processed" / source

        # Validate source
        if source not in self.SOURCES:
            logger.error(f"Unknown source: {source}. Available: {list(self.SOURCES.keys())}")
            return pd.DataFrame()

        text_column = self.SOURCES[source]

        # Load data
        input_path = data_dir / f"{ticker}.csv"
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            logger.error(f"Failed to read {input_path}: {e}")
            return pd.DataFrame()

        # Validate column exists
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in data. Available: {df.columns.tolist()}")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(df)} posts from {source} for {ticker}")

        # Prepare texts
        texts = df[text_column].fillna("").astype(str).tolist()

        # Analyze sentiment
        scores = []
        if len(texts) > 100 and self.use_financial_model:
            # Use batch processing for efficiency
            batch_size = 32
            for i in tqdm(range(0, len(texts), batch_size),
                          desc=f"Scoring {source}"):
                batch_texts = texts[i:i + batch_size]
                scores.extend(self.analyze_batch(batch_texts))
        else:
            # Process individually
            for text in tqdm(texts, desc=f"Scoring {source}"):
                scores.append(self.analyze_text(text))

        # Create scores DataFrame
        score_df = pd.DataFrame(scores)

        # Add prefix to score columns to identify source
        score_df = score_df.add_prefix(f"{source}_")

        # Combine with original data (no copy if possible)
        result_df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ticker}.csv"
        result_df.to_csv(output_path, index=False)

        logger.info(f"[✓] Scored {len(df)} {source} posts for {ticker}")
        logger.info(f"    Saved to: {output_path}")

        # Print summary statistics
        self._print_summary(score_df, source)

        return result_df

    def _print_summary(self, score_df: pd.DataFrame, source: str):
        """Print summary statistics"""
        if "compound" in score_df.columns:
            compound_col = "compound"
        elif f"{source}_compound" in score_df.columns:
            compound_col = f"{source}_compound"
        else:
            return

        mean_sent = score_df[compound_col].mean()
        pos_pct = (score_df[compound_col] > 0.05).mean() * 100
        neg_pct = (score_df[compound_col] < -0.05).mean() * 100

        logger.info(f"    Average sentiment: {mean_sent:.3f}")
        logger.info(f"    Positive (>0.05): {pos_pct:.1f}%")
        logger.info(f"    Negative (<-0.05): {neg_pct:.1f}%")


def score_multiple_sources(ticker: str, sources: List[str] = None, **kwargs):
    """Score sentiment from multiple sources"""
    if sources is None:
        sources = ["reddit"]  # Default

    analyzer = SocialSentimentAnalyzer(use_financial_model=False)  # Or True for financial model

    results = {}
    for source in sources:
        try:
            results[source] = analyzer.score_social(source, ticker, **kwargs)
        except Exception as e:
            logger.error(f"Failed to process {source} for {ticker}: {e}")

    return results


if __name__ == "__main__":
    # Example usage
    analyzer = SocialSentimentAnalyzer(use_financial_model=False)

    # Single source
    analyzer.score_social("reddit", ticker="MSFT")

    # Multiple sources
    # results = score_multiple_sources("AAPL", sources=["reddit", "twitter"])