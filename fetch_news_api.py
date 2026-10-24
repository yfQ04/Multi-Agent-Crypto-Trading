import requests
import pandas as pd
import datetime
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    print("Error: NEWS_API_KEY not found in environment variables. Please add it to your .env file.")
    exit(1)

DEFAULT_START_DATE = datetime.date(2024, 1, 1)
DEFAULT_END_DATE = datetime.date.today()

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_output_file() -> Path:
    """Store news CSV next to market data CSV files (BTC/ETH) by default."""
    data_candidates = [
        PROJECT_ROOT / "BTC.csv",
        PROJECT_ROOT / "ETH.csv",
    ]
    for data_csv in data_candidates:
        if data_csv.exists():
            return data_csv.parent / "historical_news.csv"
    return PROJECT_ROOT / "historical_news.csv"


OUTPUT_FILE = _resolve_output_file()

# Keywords related to bitcoin, crypto, and macroeconomics
QUERY = '(bitcoin OR crypto OR cryptocurrency OR "interest rate" OR "federal reserve" OR inflation OR war OR conflict)'

def fetch_news_for_date(date, page_size: int = 100, max_pages: int = 5):
    url = "https://newsapi.org/v2/everything"

    try:
        articles = []
        for page in range(1, max_pages + 1):
            params = {
                'q': QUERY,
                'from': date.strftime('%Y-%m-%d'),
                'to': date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': int(page_size),
                'page': page,
                'apiKey': NEWS_API_KEY
            }

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'ok':
                break

            page_articles = data.get('articles', [])
            for article in page_articles:
                articles.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'publishedAt': article.get('publishedAt'),
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url')
                })

            if len(page_articles) < page_size:
                break

        return articles
    except Exception as e:
        print(f"Error fetching data for {date}: {e}")
        return []


def _parse_date(value: str) -> datetime.date:
    return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch historical news into CSV")
    parser.add_argument("--start-date", default=str(DEFAULT_START_DATE), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=str(DEFAULT_END_DATE), help="YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output CSV path")
    parser.add_argument("--sleep-seconds", type=float, default=0.6, help="Sleep between days")
    parser.add_argument("--page-size", type=int, default=100, help="NewsAPI page size (max 100)")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages per day")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing CSV and refetch from start")
    return parser

def main():
    args = _build_parser().parse_args()
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if end_date < start_date:
        raise ValueError("end-date must be >= start-date")

    print(f"Fetching news from {start_date} to {end_date}...")
    print(f"Output file: {output_file}")
    
    all_articles = []
    current_date = start_date
    
    # Check if we already have partial data to resume
    if (not args.overwrite) and output_file.exists() and output_file.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(output_file)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
        if not existing_df.empty and "date" in existing_df.columns:
            last_date_str = sorted(existing_df["date"].unique())[-1]
            last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
            if last_date >= start_date:
                current_date = last_date + datetime.timedelta(days=1)
                print(f"Found existing data up to {last_date}. Resuming from {current_date}...")
                all_articles = existing_df.to_dict("records")
    
    while current_date <= end_date:
        print(f"Fetching news for {current_date}...")
        daily_articles = fetch_news_for_date(
            current_date,
            page_size=max(1, min(int(args.page_size), 100)),
            max_pages=max(1, int(args.max_pages)),
        )
        all_articles.extend(daily_articles)
        
        # Save incrementally
        df = pd.DataFrame(all_articles)
        if not df.empty:
            dedupe_cols = [c for c in ["date", "publishedAt", "title", "url"] if c in df.columns]
            if dedupe_cols:
                df = df.drop_duplicates(subset=dedupe_cols, keep="first")
        df.to_csv(output_file, index=False)
        
        current_date += datetime.timedelta(days=1)
        # Avoid hitting rate limits too hard
        time.sleep(max(0.0, float(args.sleep_seconds)))
        
    print(f"Finished fetching news. The result is saved to {output_file}")

if __name__ == "__main__":
    main()
