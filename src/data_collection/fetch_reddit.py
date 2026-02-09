import requests
import pandas as pd
from datetime import datetime
import time
import re
import os
from pathlib import Path
from urllib.parse import quote_plus

from src.utils.ticker_aliases import get_ticker_aliases

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_REDDIT = _ROOT / "data" / "reddit"
DATA_RAW_REDDIT = _ROOT / "data" / "raw" / "reddit"


def fetch_reddit_for_ticker(ticker="MSFT"):
    """Fetch Reddit posts mentioning a specific stock ticker. For symbols like BTC-USD,
    also searches for aliases (e.g. Bitcoin, btc, btc/usd) so Reddit results match
    how people refer to the asset."""

    # Normalize and get search/mention aliases (e.g. BTC-USD -> Bitcoin, BTC, btc/usd, etc.)
    ticker = ticker.upper().strip()
    aliases = get_ticker_aliases(ticker)

    # Build regex patterns for mention detection (word-boundary safe, case-insensitive)
    ticker_patterns = []
    for a in aliases:
        if not a or not str(a).strip():
            continue
        escaped = re.escape(str(a).strip())
        # Allow $ prefix for ticker symbols
        if escaped.replace("\\", "").replace("$", "").isalnum() or "/" in a or "-" in a:
            ticker_patterns.append(rf"\b{escaped}\b")
        else:
            ticker_patterns.append(rf"\b{escaped}\b")
    ticker_patterns.append(rf"\${re.escape(ticker.split('-')[0] if '-' in ticker else ticker)}\b")
    ticker_patterns = list(dict.fromkeys([p for p in ticker_patterns if p]))

    # Financial subreddits
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'options', 'stockmarket','indianstreetbets','indianstocks','indiainvestments',"cryptomarkets" ,"cryptocurrency"]

    all_posts = []
    seen_post_ids = set()

    for sub in subreddits:
        print(f"🔍 Searching r/{sub} for {ticker} (aliases: {', '.join(aliases[:5])}{'...' if len(aliases) > 5 else ''})...")

        # Try multiple search endpoints: main ticker + alias-based queries
        base = ticker.split("-")[0] if "-" in ticker else ticker
        urls_to_try = [
            f'https://www.reddit.com/r/{sub}/search.json?q={quote_plus(ticker)}&restrict_sr=on&sort=new&t=month&limit=100',
            f'https://www.reddit.com/r/{sub}/search.json?q={quote_plus(base)}&restrict_sr=on&sort=relevance&t=year&limit=100',
            f'https://www.reddit.com/r/{sub}/search.json?q=${quote_plus(base)}&restrict_sr=on&sort=relevance&t=year&limit=100',
            f'https://www.reddit.com/r/{sub}/search.json?q={quote_plus(ticker)}+stock&restrict_sr=on&sort=top&t=month&limit=100',
        ]
        for alias in aliases[:4]:
            if alias and len(str(alias)) > 2 and alias not in (ticker, base):
                urls_to_try.append(
                    f'https://www.reddit.com/r/{sub}/search.json?q={quote_plus(alias)}&restrict_sr=on&sort=new&t=month&limit=100'
                )

        for url in urls_to_try:
            try:
                response = requests.get(
                    url,
                    headers={'User-Agent': f'FINAI-Research-Ticker-{ticker}/1.0'},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    if 'data' in data and 'children' in data['data']:
                        posts_found = 0

                        for item in data['data']['children']:
                            post = item['data']
                            post_id = post.get('id')
                            if post_id in seen_post_ids:
                                continue
                            seen_post_ids.add(post_id)

                            # Extract text content
                            title = post.get('title', '')
                            content = post.get('selftext', '')
                            full_text = (title + ' ' + content).lower()
                            full_text_original = title + ' ' + content

                            # Check if ticker or any alias is mentioned (e.g. Bitcoin, btc, btc/usd for BTC-USD)
                            if not any(re.search(pattern, full_text_original, re.IGNORECASE) for pattern in ticker_patterns):
                                continue

                            sentiment = analyze_sentiment(full_text)
                            mention_count = sum(
                                len(re.findall(pattern, full_text_original, re.IGNORECASE))
                                for pattern in ticker_patterns
                            )
                            all_posts.append({
                                'ticker': ticker,
                                'subreddit': sub,
                                'title': post.get('title'),
                                'score': post.get('score', 0),
                                'upvotes': post.get('ups', 0),
                                'downvotes': post.get('downs', 0),
                                'upvote_ratio': post.get('upvote_ratio', 0),
                                'comments': post.get('num_comments', 0),
                                'created': datetime.fromtimestamp(post.get('created_utc')),
                                'author': post.get('author'),
                                'post_id': post.get('id'),
                                'url': f"https://reddit.com{post.get('permalink', '')}",
                                'content': post.get('selftext', '')[:1500],
                                'mention_count': mention_count,
                                'sentiment': sentiment,
                                'flair': post.get('link_flair_text', ''),
                                'is_original_content': post.get('is_original_content', False)
                            })
                            posts_found += 1

                        if posts_found > 0:
                            print(f"   ✅ Found {posts_found} posts in r/{sub}")

                        # Break after first successful URL
                        if posts_found > 0:
                            break

                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"   ❌ Error with r/{sub}: {e}")
                continue

    df = pd.DataFrame(all_posts)

    if not df.empty:
        # Sort by date (newest first)
        df = df.sort_values('created', ascending=False)

        # Create directory if it doesn't exist
        save_path = str(DATA_REDDIT)
        os.makedirs(save_path, exist_ok=True)

        # Save as Parquet file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}.csv"
        filepath = os.path.join(save_path, filename)

        df.to_csv(filepath, index=False)

        print(f"\n📊 Summary for {ticker}:")
        print(f"   Total posts: {len(df)}")
        print(f"   Subreddit distribution:")
        print(df['subreddit'].value_counts())
        print(f"   Date range: {df['created'].min()} to {df['created'].max()}")
        print(f"   Avg sentiment: {df['sentiment'].mean():.2f}")
        print(f"   💾 Saved to: {filepath}")

        # # Also save a small CSV sample for quick viewing
        # csv_sample_path = os.path.join(save_path, f"reddit_{ticker}_{timestamp}_sample.csv")
        # df.head(50).to_csv(csv_sample_path, index=False)
        # print(f"   📝 CSV sample saved to: {csv_sample_path}")
    else:
        print(f"\n❌ No posts found for {ticker}")

    return df, filepath if not df.empty else None


def analyze_sentiment(text):
    """Simple sentiment analysis for financial posts"""
    text_lower = text.lower()

    # Positive indicators
    positive_words = [
        'bull', 'bullish', 'buy', 'long', 'moon', 'rocket', 'growth',
        'profit', 'gain', 'win', 'success', 'good', 'great', 'strong',
        'increase', 'rise', 'surge', 'beat', 'outperform', 'undervalued'
    ]

    # Negative indicators
    negative_words = [
        'bear', 'bearish', 'sell', 'short', 'crash', 'drop', 'fall',
        'loss', 'risk', 'fail', 'bad', 'weak', 'decrease', 'decline',
        'dump', 'overvalued', 'warning', 'danger', 'trouble', 'fear'
    ]

    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)

    # Calculate simple sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total > 0:
        return (positive_count - negative_count) / total
    return 0


def fetch_ticker_mentions(ticker="MSFT", days_back=30, min_score=5):
    """More focused search for ticker mentions"""

    ticker = ticker.upper()

    # Time range
    import time as tm
    current_time = int(tm.time())
    start_time = current_time - (days_back * 24 * 60 * 60)

    # Use Pushshift for historical data (if available)
    try:
        pushshift_url = f"https://api.pushshift.io/reddit/search/submission/"
        params = {
            'q': ticker,
            'subreddit': 'stocks,investing,wallstreetbets',
            'after': start_time,
            'size': 500,
            'sort': 'desc',
            'sort_type': 'score'
        }

        response = requests.get(pushshift_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()['data']

            posts = []
            for post in data:
                if post.get('score', 0) >= min_score:
                    title = post.get('title', '')
                    body = post.get('selftext', '')

                    # Verify it's actually about the ticker
                    if (ticker in title.upper() or
                            f'${ticker}' in title or
                            ticker in body.upper()):
                        posts.append({
                            'ticker': ticker,
                            'title': title,
                            'score': post.get('score'),
                            'comments': post.get('num_comments', 0),
                            'created': datetime.fromtimestamp(post.get('created_utc')),
                            'subreddit': post.get('subreddit'),
                            'url': post.get('full_link', ''),
                            'content': body[:1000]
                        })

            if posts:
                df = pd.DataFrame(posts)
                df = df.sort_values('score', ascending=False)

                # Save to Parquet
                save_path = str(DATA_RAW_REDDIT)
                os.makedirs(save_path, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{ticker}.csv"
                filepath = os.path.join(save_path, filename)

                df.to_csv(filepath, index=False)
                print(f"✅ Found {len(df)} historical posts for {ticker} (Pushshift)")
                print(f"💾 Saved to: {filepath}")
                return df, filepath
    except Exception as e:
        print(f"Pushshift error (using Reddit API instead): {e}")

    # Fallback to Reddit API
    return fetch_reddit_for_ticker(ticker)


# def read_parquet_file(filepath):
#     """Helper function to read and display parquet file"""
#     try:
#         df = pd.read_parquet(filepath)
#         print(f"\n📖 Reading from {filepath}")
#         print(f"   Shape: {df.shape}")
#         print(f"   Columns: {', '.join(df.columns)}")
#         return df
#     except Exception as e:
#         print(f"❌ Error reading parquet file: {e}")
#         return None


if __name__ == "__main__":
    # Fetch posts for AAPL
    ticker = "MSFT"  # Change this to any ticker you want
    print(f"🚀 Starting Reddit data collection for {ticker}...")
    print("=" * 60)

    # Create save directory
    save_dir = str(DATA_REDDIT)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Save directory: {save_dir}")

    # Fetch recent data
    data, filepath = fetch_reddit_for_ticker(ticker)

    # if not data.empty:
    #     print(f"\n📈 Top 5 posts for {ticker}:")
    #     for idx, row in data.head().iterrows():
    #         print(f"\n{idx + 1}. [{row['subreddit']}] {row['title'][:80]}...")
    #         print(f"   ⭐ Score: {row['score']} | 💬 Comments: {row['comments']}")
    #         print(f"   📅 {row['created'].strftime('%Y-%m-%d %H:%M')}")
    #         print(f"   Sentiment: {row['sentiment']:.2f}")
    #
    #     # Verify the parquet file was saved correctly
    #     if filepath:
    #         print(f"\n🔍 Verifying saved file...")
    #         df_check = read_parquet_file(filepath)
    #         if df_check is not None:
    #             print("✅ Parquet file verified successfully!")
    #
    #     # Optional: Also try historical data
    #     print("\n" + "=" * 60)
    #     print("📚 Attempting to fetch historical data...")
    #     historical_data, historical_filepath = fetch_ticker_mentions(ticker, days_back=30)
    #
    #     if historical_data is not None and not historical_data.empty:
    #         print(f"✅ Total data collected: {len(data) + len(historical_data)} posts")
    #
    #         # Combine both datasets
    #         combined_df = pd.concat([data, historical_data], ignore_index=True)
    #         combined_df = combined_df.drop_duplicates(subset=['post_id'], keep='first')
    #
    #         # Save combined data
    #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #         combined_filename = f"reddit_{ticker}_combined_{timestamp}.parquet"
    #         combined_filepath = os.path.join(save_dir, combined_filename)
    #         combined_df.to_parquet(combined_filepath, index=False)
    #
    #         print(f"\n📊 Combined dataset:")
    #         print(f"   Unique posts: {len(combined_df)}")
    #         print(f"   Saved to: {combined_filepath}")
    # else:
    #     print(f"\n⚠ Try a different approach or check if {ticker} is being discussed")
# # save as fetch_reddit_aapl.py
# import requests
# import pandas as pd
# from datetime import datetime
# import time
# import re
# import os
#
#
# def get_ticker_aliases(ticker: str):
#     """
#     Expand a trading ticker into common Reddit-friendly aliases.
#     """
#     base = ticker.upper()
#
#     aliases = set()
#
#     # Handle crypto pairs like SOL-USD, BTC-USD
#     if "-" in base:
#         asset = base.split("-")[0]
#         aliases.update({
#             asset,
#             f"${asset}",
#             base,
#             base.replace("-", "/"),
#         })
#
#         # Hard-coded common crypto names (extendable)
#         crypto_names = {
#             "BTC": "Bitcoin",
#             "ETH": "Ethereum",
#             "SOL": "Solana",
#             "ADA": "Cardano",
#             "XRP": "Ripple",
#             "DOGE": "Dogecoin",
#             "AVAX": "Avalanche",
#             "XAU": "Gold",
#             "XAG": "Silver",
#             "MSFT": "Microsoft",
#             "AAPL": "Apple",
#             "GOOGL": "Google",
#             "AMZN": "Amazon",
#             "TSLA": "Tesla",
#             "NVDA": "NVIDIA",
#             "META": "Meta",
#             "NFLX": "Netflix"
#         }
#
#         if asset in crypto_names:
#             aliases.add(crypto_names[asset])
#
#     else:
#         aliases.update({
#             base,
#             f"${base}"
#         })
#
#         # Add company name for major stocks
#         stock_names = {
#             "MSFT": ["Microsoft", "MSFT", "$MSFT"],
#             "AAPL": ["Apple", "AAPL", "$AAPL", "iphone", "macbook"],
#             "GOOGL": ["Google", "Alphabet", "GOOGL", "$GOOGL", "$GOOG"],
#             "AMZN": ["Amazon", "AMZN", "$AMZN"],
#             "TSLA": ["Tesla", "TSLA", "$TSLA", "Elon Musk"],
#             "NVDA": ["NVIDIA", "NVDA", "$NVDA"],
#             "META": ["Meta", "Facebook", "META", "$META"],
#             "NFLX": ["Netflix", "NFLX", "$NFLX"],
#             "XAU": ["Gold", "XAU", "$XAU", "gold price"],
#             "XAG": ["Silver", "XAG", "$XAG", "silver price"]
#         }
#
#         if base in stock_names:
#             aliases.update(stock_names[base])
#
#     return list(aliases)
#
#
# def fetch_reddit_for_ticker(ticker="MSFT"):
#     """Fetch Reddit posts mentioning a specific stock ticker using aliases"""
#
#     # Get all possible aliases for the ticker
#     aliases = get_ticker_aliases(ticker)
#     print(f"🔍 Searching for {ticker} with aliases: {', '.join(aliases)}")
#
#     # Convert to uppercase and add $ symbol for better search
#     ticker_base = ticker.upper().split('-')[0] if '-' in ticker else ticker.upper()
#
#     # Financial subreddits
#     subreddits = ['stocks', 'investing', 'wallstreetbets', 'options', 'stockmarket',
#                   'indianstreetbets', 'indianstocks', 'indiainvestments',
#                   "cryptomarkets", "cryptocurrency", "CryptoCurrency"]
#
#     all_posts = []
#
#     for sub in subreddits:
#         print(f"📊 Searching r/{sub}...")
#
#         # Try multiple search queries using all aliases
#         search_terms = [
#             f'{ticker_base}',
#             f'${ticker_base}',
#             f'#{ticker_base}'
#         ]
#
#         # Add company names if available
#         if ticker_base in ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']:
#             search_terms.append(ticker_base.lower())  # For lowercase mentions
#             if ticker_base == 'MSFT':
#                 search_terms.extend(['microsoft', 'Microsoft'])
#             elif ticker_base == 'AAPL':
#                 search_terms.extend(['apple', 'Apple', 'iphone'])
#             elif ticker_base == 'XAU':
#                 search_terms.extend(['gold', 'Gold'])
#             elif ticker_base == 'XAG':
#                 search_terms.extend(['silver', 'Silver'])
#
#         for search_term in search_terms:
#             urls_to_try = [
#                 f'https://www.reddit.com/r/{sub}/search.json?q={search_term}&restrict_sr=on&sort=new&t=month&limit=100',
#                 f'https://www.reddit.com/r/{sub}/search.json?q={search_term}+stock&restrict_sr=on&sort=relevance&t=year&limit=100',
#                 f'https://www.reddit.com/r/{sub}/search.json?q={search_term}+price&restrict_sr=on&sort=top&t=month&limit=100'
#             ]
#
#             for url in urls_to_try:
#                 try:
#                     response = requests.get(
#                         url,
#                         headers={'User-Agent': f'FINAI-Research-Ticker-{ticker_base}/1.0'},
#                         timeout=30
#                     )
#
#                     if response.status_code == 200:
#                         data = response.json()
#
#                         if 'data' in data and 'children' in data['data']:
#                             posts_found = 0
#
#                             for item in data['data']['children']:
#                                 post = item['data']
#
#                                 # Extract text content
#                                 title = post.get('title', '').lower()
#                                 content = post.get('selftext', '').lower()
#                                 full_text = title + ' ' + content
#
#                                 # Check if any alias is mentioned
#                                 match_found = False
#                                 mention_count = 0
#
#                                 for alias in aliases:
#                                     alias_lower = alias.lower()
#                                     # Check for exact word boundaries
#                                     patterns = [
#                                         rf'\b{alias_lower}\b',
#                                         rf'\${alias_lower}\b',
#                                         rf'\#{alias_lower}\b'
#                                     ]
#
#                                     for pattern in patterns:
#                                         matches = re.findall(pattern, full_text, re.IGNORECASE)
#                                         if matches:
#                                             match_found = True
#                                             mention_count += len(matches)
#                                             break
#
#                                     if match_found:
#                                         break
#
#                                 if match_found:
#                                     # Extract sentiment indicators
#                                     sentiment = analyze_sentiment(full_text)
#
#                                     all_posts.append({
#                                         'ticker': ticker,
#                                         'ticker_base': ticker_base,
#                                         'subreddit': sub,
#                                         'title': post.get('title'),
#                                         'score': post.get('score', 0),
#                                         'upvotes': post.get('ups', 0),
#                                         'downvotes': post.get('downs', 0),
#                                         'upvote_ratio': post.get('upvote_ratio', 0),
#                                         'comments': post.get('num_comments', 0),
#                                         'created': datetime.fromtimestamp(post.get('created_utc')),
#                                         'author': post.get('author'),
#                                         'post_id': post.get('id'),
#                                         'url': f"https://reddit.com{post.get('permalink', '')}",
#                                         'content': post.get('selftext', '')[:1500],
#                                         'mention_count': mention_count,
#                                         'sentiment': sentiment,
#                                         'flair': post.get('link_flair_text', ''),
#                                         'is_original_content': post.get('is_original_content', False),
#                                         'search_term_matched': search_term
#                                     })
#                                     posts_found += 1
#
#                             if posts_found > 0:
#                                 print(f"   ✅ Found {posts_found} posts in r/{sub} using '{search_term}'")
#
#                             # Break after first successful URL for this search term
#                             if posts_found > 0:
#                                 break
#
#                     time.sleep(1)  # Rate limiting
#
#                 except Exception as e:
#                     print(f"   ❌ Error with r/{sub} search term '{search_term}': {e}")
#                     continue
#
#             # Short pause between search terms
#             time.sleep(0.5)
#
#     # Convert to DataFrame
#     if all_posts:
#         df = pd.DataFrame(all_posts)
#
#         # Remove duplicates based on post_id
#         df = df.drop_duplicates(subset=['post_id'], keep='first')
#
#         # Sort by date (newest first)
#         df = df.sort_values('created', ascending=False)
#
#         # Create directory if it doesn't exist
#         save_path = r'C:\Users\SHREEL\PycharmProjects\FINAI\data\reddit'
#         os.makedirs(save_path, exist_ok=True)
#
#         # Save as CSV file
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"{ticker_base}_{timestamp}.csv"
#         filepath = os.path.join(save_path, filename)
#
#         df.to_csv(filepath, index=False)
#
#         print(f"\n📊 Summary for {ticker}:")
#         print(f"   Total unique posts: {len(df)}")
#         print(f"   Subreddit distribution:")
#         print(df['subreddit'].value_counts().head())
#         print(f"   Date range: {df['created'].min()} to {df['created'].max()}")
#         print(f"   Avg sentiment: {df['sentiment'].mean():.2f}")
#         print(f"   Total mentions: {df['mention_count'].sum()}")
#         print(f"   💾 Saved to: {filepath}")
#
#         return df, filepath
#     else:
#         print(f"\n❌ No posts found for {ticker} with any alias")
#         return pd.DataFrame(), None
#
#
# def analyze_sentiment(text):
#     """Simple sentiment analysis for financial posts"""
#     text_lower = text.lower()
#
#     # Positive indicators
#     positive_words = [
#         'bull', 'bullish', 'buy', 'long', 'moon', 'rocket', 'growth',
#         'profit', 'gain', 'win', 'success', 'good', 'great', 'strong',
#         'increase', 'rise', 'surge', 'beat', 'outperform', 'undervalued',
#         'up', 'rally', 'breakout', 'support', 'positive', 'optimistic',
#         'opportunity', 'recovery', 'rebound'
#     ]
#
#     # Negative indicators
#     negative_words = [
#         'bear', 'bearish', 'sell', 'short', 'crash', 'drop', 'fall',
#         'loss', 'risk', 'fail', 'bad', 'weak', 'decrease', 'decline',
#         'dump', 'overvalued', 'warning', 'danger', 'trouble', 'fear',
#         'down', 'collapse', 'resistance', 'negative', 'pessimistic',
#         'warning', 'caution', 'danger', 'avoid'
#     ]
#
#     positive_count = sum(text_lower.count(word) for word in positive_words)
#     negative_count = sum(text_lower.count(word) for word in negative_words)
#
#     # Calculate simple sentiment score (-1 to 1)
#     total = positive_count + negative_count
#     if total > 0:
#         return (positive_count - negative_count) / total
#     return 0
#
#
# def fetch_ticker_mentions(ticker="MSFT", days_back=30, min_score=5):
#     """More focused search for ticker mentions using aliases"""
#
#     aliases = get_ticker_aliases(ticker)
#     ticker_base = ticker.upper().split('-')[0] if '-' in ticker else ticker.upper()
#
#     print(f"📚 Fetching historical mentions for {ticker} with aliases: {', '.join(aliases)}")
#
#     # Time range
#     import time as tm
#     current_time = int(tm.time())
#     start_time = current_time - (days_back * 24 * 60 * 60)
#
#     all_posts = []
#
#     # Try each alias with Pushshift
#     for alias in aliases:
#         if len(str(alias)) < 2:  # Skip very short aliases
#             continue
#
#         try:
#             pushshift_url = "https://api.pushshift.io/reddit/search/submission/"
#             params = {
#                 'q': alias,
#                 'subreddit': 'stocks,investing,wallstreetbets,cryptocurrency,CryptoCurrency',
#                 'after': start_time,
#                 'size': 200,
#                 'sort': 'desc',
#                 'sort_type': 'score'
#             }
#
#             response = requests.get(pushshift_url, params=params, timeout=30)
#             if response.status_code == 200:
#                 data = response.json().get('data', [])
#
#                 for post in data:
#                     if post.get('score', 0) >= min_score:
#                         title = post.get('title', '').lower()
#                         body = post.get('selftext', '').lower()
#                         full_text = title + ' ' + body
#
#                         # Verify it's actually about our ticker using all aliases
#                         alias_match = False
#                         for check_alias in aliases:
#                             check_alias_lower = str(check_alias).lower()
#                             patterns = [
#                                 rf'\b{check_alias_lower}\b',
#                                 rf'\${check_alias_lower}\b',
#                                 rf'\#{check_alias_lower}\b'
#                             ]
#
#                             for pattern in patterns:
#                                 if re.search(pattern, full_text, re.IGNORECASE):
#                                     alias_match = True
#                                     break
#
#                             if alias_match:
#                                 break
#
#                         if alias_match:
#                             sentiment = analyze_sentiment(full_text)
#
#                             all_posts.append({
#                                 'ticker': ticker,
#                                 'ticker_base': ticker_base,
#                                 'title': post.get('title'),
#                                 'score': post.get('score', 0),
#                                 'comments': post.get('num_comments', 0),
#                                 'created': datetime.fromtimestamp(post.get('created_utc')),
#                                 'subreddit': post.get('subreddit'),
#                                 'url': post.get('full_link', ''),
#                                 'content': post.get('selftext', '')[:1000],
#                                 'sentiment': sentiment,
#                                 'alias_matched': alias,
#                                 'source': 'pushshift'
#                             })
#
#                 print(
#                     f"   ✅ Found {len([p for p in all_posts if p['alias_matched'] == alias])} posts with alias '{alias}'")
#
#             time.sleep(1)  # Rate limiting for Pushshift
#
#         except Exception as e:
#             print(f"   ❌ Error with alias '{alias}': {e}")
#             continue
#
#     if all_posts:
#         df = pd.DataFrame(all_posts)
#         # Remove duplicates based on URL
#         df = df.drop_duplicates(subset=['url'], keep='first')
#         df = df.sort_values('created', ascending=False)
#
#         # Save to CSV
#         save_path = r'C:\Users\SHREEL\PycharmProjects\FINAI\data\reddit'
#         os.makedirs(save_path, exist_ok=True)
#
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = f"{ticker_base}_historical_{timestamp}.csv"
#         filepath = os.path.join(save_path, filename)
#
#         df.to_csv(filepath, index=False)
#         print(f"\n✅ Found {len(df)} historical posts for {ticker}")
#         print(f"💾 Saved to: {filepath}")
#         return df, filepath
#
#     print(f"⚠ No historical posts found for {ticker}")
#     return pd.DataFrame(), None
#
#
# # if __name__ == "__main__":
# #     # Test with different tickers
# #     test_tickers = ["MSFT", "XAU-USD", "BTC-USD", "AAPL"]
# #
# #     for ticker in test_tickers:
# #         print(f"\n{'=' * 60}")
# #         print(f"🚀 Starting Reddit data collection for {ticker}...")
# #         print(f"{'=' * 60}")
# #
# #         # Create save directory
# #         save_dir = r'C:\Users\SHREEL\PycharmProjects\FINAI\data\reddit'
# #         os.makedirs(save_dir, exist_ok=True)
# #
# #         # Test alias function
# #         aliases = get_ticker_aliases(ticker)
# #         print(f"📋 Aliases for {ticker}: {aliases}")
# #
# #         # Fetch recent data
# #         data, filepath = fetch_reddit_for_ticker(ticker)
# #
# #         if not data.empty:
# #             print(f"\n📈 Sample posts for {ticker}:")
# #             for idx, row in data.head(3).iterrows():
# #                 print(f"\n{idx + 1}. [{row['subreddit']}] {row['title'][:80]}...")
# #                 print(f"   ⭐ Score: {row['score']} | 💬 Comments: {row['comments']}")
# #                 print(f"   📅 {row['created'].strftime('%Y-%m-%d %H:%M')}")
# #                 print(f"   Sentiment: {row['sentiment']:.2f} | Mentions: {row['mention_count']}")
# #
# #             # Also try historical data
# #             print(f"\n📚 Fetching historical data for {ticker}...")
# #             historical_data, historical_filepath = fetch_ticker_mentions(ticker, days_back=14)
# #
# #             if not historical_data.empty:
# #                 # Combine both datasets
# #                 combined_df = pd.concat([data, historical_data], ignore_index=True)
# #                 combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
# #
# #                 # Save combined data
# #                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# #                 combined_filename = f"{ticker.replace('-', '_')}_combined_{timestamp}.csv"
# #                 combined_filepath = os.path.join(save_dir, combined_filename)
# #                 combined_df.to_csv(combined_filepath, index=False)
# #
# #                 print(f"\n📊 Combined dataset for {ticker}:")
# #                 print(f"   Unique posts: {len(combined_df)}")
# #                 print(f"   Date range: {combined_df['created'].min()} to {combined_df['created'].max()}")
# #                 print(f"   Avg sentiment: {combined_df['sentiment'].mean():.2f}")
# #                 print(f"   💾 Saved to: {combined_filepath}")
# #         else:
# #             print(f"\n⚠ No data found for {ticker}. Try a different ticker or check if it's being discussed.")
# #
# #         print(f"\n✅ Completed processing for {ticker}")
# #         time.sleep(5)  # Wait between tickers to avoid rate limiting