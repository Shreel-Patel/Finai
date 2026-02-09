import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging
from typing import Optional, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailySentimentAggregator:
    """Aggregate sentiment data to daily frequency"""

    def __init__(self):
        # Define date columns for different sources
        self.DATE_COLUMNS = {
            "reddit": "created",
            "twitter": "created_at",
            "news": "pubDate",
            "stocktwits": "created_at",
            "default": "date"  # Fallback
        }

        # Define sentiment columns for different sources
        self.SENTIMENT_COLUMNS = {
            "reddit": "compound",
            "twitter": "compound",
            "news": "compound",  # Or "sentiment_score"
            "default": "sentiment"
        }

    def aggregate_daily(
            self,
            source: str,
            ticker: str,
            weight_col: Optional[str] = None,
            data_dir: Optional[Path] = None,
            output_dir: Optional[Path] = None,
            fill_missing_days: bool = True,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate sentiment data to daily frequency

        Args:
            source: Data source (reddit, twitter, news, etc.)
            ticker: Stock ticker symbol
            weight_col: Column to use for weighting (e.g., 'score', 'upvotes', 'volume')
            data_dir: Directory containing processed data
            output_dir: Directory for output features
            fill_missing_days: If True, fill missing dates with neutral sentiment
            start_date: Start date for date range (YYYY-MM-DD)
            end_date: End date for date range (YYYY-MM-DD)

        Returns:
            DataFrame with daily sentiment metrics
        """
        # Setup paths
        if data_dir is None:
            data_dir = Path(f"C:\\Users\\SHREEL\\PycharmProjects\\FINAI\\data\\processed\\{source}")

        if output_dir is None:
            output_dir = Path("C:\\Users\\SHREEL\\PycharmProjects\\FINAI\\data\\features\\sentiment")

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

        logger.info(f"Aggregating {len(df)} records from {source} for {ticker}")

        # 1. Parse dates
        df = self._parse_dates(df, source)
        if df is None or df.empty:
            return pd.DataFrame()

        # 2. Get sentiment column
        sentiment_col = self._get_sentiment_column(df, source)
        if sentiment_col is None:
            return pd.DataFrame()

        # 3. Calculate weighted sentiment
        df = self._calculate_weighted_sentiment(df, sentiment_col, weight_col)

        # 4. Aggregate to daily frequency
        daily_df = self._aggregate_daily_metrics(df, sentiment_col)

        # 5. Handle date range
        daily_df = self._handle_date_range(
            daily_df,
            fill_missing_days=fill_missing_days,
            start_date=start_date,
            end_date=end_date
        )

        # 6. Add rolling statistics
        daily_df = self._add_rolling_metrics(daily_df)

        # 7. Save results
        output_path = output_dir / f"{source}_{ticker}_daily.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_path, index=False)

        logger.info(f"[✓] Aggregated {len(daily_df)} daily records for {ticker}")
        logger.info(f"    Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        logger.info(f"    Saved to: {output_path}")

        # 7b. For reddit, also write reddit_{ticker}.csv (no _daily) so build_dataset can read it
        if source == "reddit" and "sentiment_mean" in daily_df.columns:
            simple_path = output_dir / f"reddit_{ticker}.csv"
            simple_df = daily_df[["date"]].copy()
            simple_df["compound"] = daily_df["sentiment_mean"].values
            if "weighted_sentiment_mean" in daily_df.columns:
                simple_df["weighted_sentiment"] = daily_df["weighted_sentiment_mean"].values
            simple_df.to_csv(simple_path, index=False)
            logger.info(f"    Also saved build_dataset input: {simple_path}")

        # Print summary
        self._print_summary(daily_df)

        return daily_df

    def _parse_dates(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Parse and validate date columns"""
        # Get date column name
        date_col = self.DATE_COLUMNS.get(source, self.DATE_COLUMNS["default"])

        # Check if column exists
        if date_col not in df.columns:
            logger.error(f"Date column '{date_col}' not found. Available columns: {df.columns.tolist()}")
            # Try to find any date-like column
            date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_candidates:
                date_col = date_candidates[0]
                logger.info(f"Using alternative date column: {date_col}")
            else:
                return pd.DataFrame()

        # Parse dates (support ISO8601 e.g. "2025-01-30 12:00:00+00:00")
        try:
            df['date'] = pd.to_datetime(df[date_col], format="ISO8601", utc=True).dt.date
        except (ValueError, TypeError):
            try:
                df['date'] = pd.to_datetime(df[date_col], utc=True).dt.date
            except Exception as e:
                logger.error(f"Failed to parse dates from column '{date_col}': {e}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to parse dates from column '{date_col}': {e}")
            return pd.DataFrame()

        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])

        return df

    def _get_sentiment_column(self, df: pd.DataFrame, source: str) -> Optional[str]:
        """Get the appropriate sentiment column"""
        # Try source-specific column
        sentiment_col = self.SENTIMENT_COLUMNS.get(source)
        if sentiment_col and sentiment_col in df.columns:
            return sentiment_col

        # Try common sentiment columns
        sentiment_candidates = [
            'compound', 'sentiment', 'sentiment_score',
            'weighted_sentiment', 'pos', 'neg'
        ]

        for col in sentiment_candidates:
            if col in df.columns:
                logger.info(f"Using sentiment column: {col}")
                return col

        logger.error(f"No sentiment column found. Available: {df.columns.tolist()}")
        return None

    def _calculate_weighted_sentiment(self, df: pd.DataFrame, sentiment_col: str,
                                      weight_col: Optional[str]) -> pd.DataFrame:
        """Calculate weighted sentiment scores"""
        if weight_col and weight_col in df.columns:
            # Normalize weights (0-1 range)
            if df[weight_col].max() > 0:
                normalized_weights = df[weight_col] / df[weight_col].max()
            else:
                normalized_weights = 1

            # Calculate weighted sentiment
            df['weighted_sentiment'] = df[sentiment_col] * normalized_weights

            # Also calculate sentiment magnitude (absolute value weighted)
            df['sentiment_magnitude'] = df[sentiment_col].abs() * normalized_weights
        else:
            df['weighted_sentiment'] = df[sentiment_col]
            df['sentiment_magnitude'] = df[sentiment_col].abs()

        return df

    def _aggregate_daily_metrics(self, df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """Aggregate metrics to daily frequency"""

        aggregation_rules = {
            # Basic sentiment metrics
            sentiment_col: ['mean', 'std', 'count', lambda x: (x > 0.05).sum(), lambda x: (x < -0.05).sum()],
            'weighted_sentiment': ['mean', 'std'],
            'sentiment_magnitude': ['mean', 'sum'],

            # Volume metrics
            'date': 'count'  # Total posts/comments
        }

        # Rename aggregation results
        daily_df = df.groupby('date').agg(aggregation_rules)
        daily_df.columns = [
            'sentiment_mean', 'sentiment_std', 'total_posts',
            'positive_posts', 'negative_posts',
            'weighted_sentiment_mean', 'weighted_sentiment_std',
            'sentiment_magnitude_mean', 'sentiment_magnitude_sum',
            'post_count'  # Same as total_posts, but keeping for clarity
        ]

        daily_df = daily_df.reset_index()

        # Calculate derived metrics
        daily_df['sentiment_variance'] = daily_df['sentiment_std'] ** 2
        daily_df['pos_neg_ratio'] = (daily_df['positive_posts'] + 1) / (daily_df['negative_posts'] + 1)
        daily_df['sentiment_volatility'] = daily_df['sentiment_std'] / (
                    daily_df['sentiment_mean'].abs() + 0.001)  # Avoid division by zero

        # Sentiment score (normalized -1 to 1)
        daily_df['sentiment_score'] = daily_df['weighted_sentiment_mean'] / (
                    daily_df['sentiment_magnitude_mean'] + 0.001)

        return daily_df

    def _handle_date_range(self, daily_df: pd.DataFrame, fill_missing_days: bool,
                           start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Handle date range and fill missing days if needed"""

        # Convert to datetime for date operations
        daily_df['date'] = pd.to_datetime(daily_df['date'])

        # Define date range
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = daily_df['date'].min()

        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = daily_df['date'].max()

        # Create complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        if fill_missing_days:
            # Reindex to complete date range
            daily_df = daily_df.set_index('date').reindex(date_range)
            daily_df.index.name = 'date'
            daily_df = daily_df.reset_index()

            # Fill missing values
            fill_values = {
                'sentiment_mean': 0,
                'sentiment_std': 0,
                'total_posts': 0,
                'positive_posts': 0,
                'negative_posts': 0,
                'weighted_sentiment_mean': 0,
                'weighted_sentiment_std': 0,
                'sentiment_magnitude_mean': 0,
                'sentiment_magnitude_sum': 0,
                'post_count': 0,
                'sentiment_variance': 0,
                'pos_neg_ratio': 1,  # Neutral ratio
                'sentiment_volatility': 0,
                'sentiment_score': 0
            }

            daily_df = daily_df.fillna(fill_values)

        return daily_df

    def _add_rolling_metrics(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""

        # Sort by date
        daily_df = daily_df.sort_values('date')

        # Rolling windows
        windows = [3, 7, 14]

        for window in windows:
            # Rolling sentiment mean
            daily_df[f'sentiment_{window}d_mean'] = daily_df['weighted_sentiment_mean'].rolling(window=window,
                                                                                                min_periods=1).mean()

            # Rolling sentiment std
            daily_df[f'sentiment_{window}d_std'] = daily_df['weighted_sentiment_mean'].rolling(window=window,
                                                                                               min_periods=1).std()

            # Rolling volume
            daily_df[f'post_count_{window}d_mean'] = daily_df['post_count'].rolling(window=window, min_periods=1).mean()

            # Sentiment change
            daily_df[f'sentiment_{window}d_change'] = daily_df['weighted_sentiment_mean'] - daily_df[
                f'sentiment_{window}d_mean']

        return daily_df

    def _print_summary(self, daily_df: pd.DataFrame):
        """Print summary statistics"""
        if daily_df.empty:
            return

        logger.info("Daily Aggregation Summary:")
        logger.info(f"  Total days: {len(daily_df)}")
        logger.info(f"  Avg daily posts: {daily_df['post_count'].mean():.1f}")
        logger.info(f"  Avg sentiment: {daily_df['sentiment_mean'].mean():.3f}")
        logger.info(f"  Days with posts: {(daily_df['post_count'] > 0).sum()}")
        logger.info(f"  Most positive day: {daily_df.loc[daily_df['sentiment_mean'].idxmax(), 'date'].date()} "
                    f"({daily_df['sentiment_mean'].max():.3f})")
        logger.info(f"  Most negative day: {daily_df.loc[daily_df['sentiment_mean'].idxmin(), 'date'].date()} "
                    f"({daily_df['sentiment_mean'].min():.3f})")


def aggregate_multiple_sources(ticker: str, sources: List[str] = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """Aggregate multiple sources for a ticker"""
    if sources is None:
        sources = ["reddit", "news"]  # Default sources

    aggregator = DailySentimentAggregator()
    results = {}

    for source in sources:
        try:
            logger.info(f"\nAggregating {source} data for {ticker}...")
            results[source] = aggregator.aggregate_daily(source, ticker, **kwargs)
        except Exception as e:
            logger.error(f"Failed to aggregate {source} for {ticker}: {e}")

    return results


if __name__ == "__main__":
    # Example usage

    # Single source
    aggregator = DailySentimentAggregator()
    daily_reddit = aggregator.aggregate_daily(
        source="reddit",
        ticker="AAPL",
        weight_col="score",
        fill_missing_days=True,
        start_date="2025-01-01",
        end_date="2026-01-31"
    )

    # Multiple sources
    # results = aggregate_multiple_sources(
    #     ticker="AAPL",
    #     sources=["reddit", "news", "twitter"],
    #     fill_missing_days=True
    # )