# agents/news_sentiment_agent.py
"""
Agent 1: News and Social Media Collection + Sentiment Analysis
- Fetch Bitcoin news from NewsAPI
- Fetch posts from Reddit (r/Bitcoin, r/CryptoCurrency)
- Perform sentiment analysis using a language model (SiliconFlow LLM)
"""

import os
import json
import hashlib
import requests
import praw
from typing import Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from agents.llm_utils import ask_llm



PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_day_key(day_str: str | None) -> str | None:
    if not day_str:
        return None
    ts = pd.to_datetime(day_str, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return str(ts.date())


def _iter_day_range(start_day: str, end_day: str) -> list[str]:
    start_ts = pd.to_datetime(start_day, errors="coerce")
    end_ts = pd.to_datetime(end_day, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return [str(d.date()) for d in pd.date_range(start=start_ts, end=end_ts, freq="D")]


def _extract_item_day(item: dict) -> str:
    ts = pd.to_datetime(str(item.get("published_at", "") or ""), errors="coerce", utc=True)
    return str(ts.date()) if pd.notna(ts) else "unknown_day"


def _get_day_results_map(cache_payload: dict) -> dict:
    raw = cache_payload.get("__day_results__")
    return raw if isinstance(raw, dict) else {}


def _save_day_results_to_cache(cache_path: Path, items: list[dict]) -> None:
    if not items:
        return
    payload = _load_sentiment_cache(cache_path)
    day_results = _get_day_results_map(payload)

    grouped: dict[str, list[dict]] = {}
    for item in items:
        day_key = _extract_item_day(item)
        grouped.setdefault(day_key, []).append(item)

    for day_key, day_items in grouped.items():
        day_results[day_key] = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "items": day_items,
        }

    payload["__day_results__"] = day_results
    _save_sentiment_cache(cache_path, payload)


def _load_cached_items_by_days(cache_path: Path, target_days: list[str]) -> tuple[list[dict], list[str], list[str]]:
    payload = _load_sentiment_cache(cache_path)
    day_results = _get_day_results_map(payload)

    hit_days: list[str] = []
    miss_days: list[str] = []
    items: list[dict] = []

    for day in target_days:
        day_payload = day_results.get(day)
        if not isinstance(day_payload, dict):
            miss_days.append(day)
            continue

        day_items = day_payload.get("items")
        if not isinstance(day_items, list) or not day_items:
            miss_days.append(day)
            continue

        hit_days.append(day)
        items.extend(day_items)

    return items, hit_days, miss_days


def _load_sentiment_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_sentiment_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _sentiment_cache_key(day_key: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{day_key}:{digest}"


def fetch_news(days: int = 7) -> list[dict]:
    """Fetch recent Bitcoin news from NewsAPI"""
    api_key = os.getenv("NEWS_API_KEY")
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q=bitcoin+OR+BTC+cryptocurrency"
        f"&from={from_date}"
        f"&sortBy=publishedAt"
        f"&language=en"
        f"&pageSize=30"
        f"&apiKey={api_key}"
    )
    resp = requests.get(url, timeout=15)
    articles = resp.json().get("articles", [])
    results = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '') or ''}"
        results.append({
            "source": "news",
            "title": a.get("title", ""),
            "text": text.strip(),
            "published_at": a.get("publishedAt", ""),
        })
    print(f"[Agent1] Retrieved {len(results)} news articles")
    return results


def _parse_llm_json(response_text: str) -> list | None:
    """
    Clean and parse LLM response as a JSON array.
    Returns the parsed list or None on failure.
    """
    clean = response_text.strip()

    # Strip markdown code fences
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    # Sometimes the model wraps in an outer object, try to extract the array
    if clean.startswith("{"):
        # Try finding an array inside
        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]

    data = json.loads(clean)
    if isinstance(data, list):
        return data
    return None


def llm_sentiment_analysis(
    texts: list[str],
    published_ats: list[str] | None = None,
    per_day_batch_size: int = 10,
    use_cache: bool = True,
    cache_file: str = "artifacts/sentiment_cache.json",
    refresh_existing: bool = False,
    on_day_complete: Callable[[str, list[int], list[dict]], None] | None = None,
) -> list[dict]:
    """
    Perform sentiment analysis using SiliconFlow LLM with per-day batching.
    Each batch processes at most 10 items. Remaining items form the last batch.
    e.g. 99 items → 9 batches of 10 + 1 batch of 9.
    """
    import time

    # ------------------------------------------------------------------ #
    # 🔒 Hard-cap: each batch is at most 10 items to avoid JSON truncation #
    # ------------------------------------------------------------------ #
    effective_batch_size = min(max(1, int(per_day_batch_size)), 5)

    # Initialize all with neutral defaults
    results = [
        {"label": "neutral_info", "score": 0.0, "confidence": 0.0, "reason": ""}
        for _ in range(len(texts))
    ]

    # Filter valid texts to save API tokens and avoid errors
    valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) >= 5]
    if not valid_indices:
        return results

    cache_path = Path(cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    sentiment_cache = _load_sentiment_cache(cache_path) if use_cache else {}
    cache_updated = False

    # ------------------------------------------------------------------ #
    # Group valid items by calendar day (UTC)                             #
    # ------------------------------------------------------------------ #
    if published_ats is None or len(published_ats) != len(texts):
        day_series = pd.Series([pd.NaT] * len(texts))
    else:
        # Parse with UTC to avoid incorrect day assignment for naive timestamps.
        day_series = pd.to_datetime(pd.Series(published_ats), errors="coerce", utc=True)

    grouped_indices: dict[str, list[int]] = {}
    invalid_indices: dict[str, int] = {}
    for idx in range(len(texts)):
        ts = day_series.iloc[idx] if idx < len(day_series) else pd.NaT
        day_key = str(ts.date()) if pd.notna(ts) else "unknown_day"
        if idx not in valid_indices:
            invalid_indices[day_key] = invalid_indices.get(day_key, 0) + 1
        else:
            grouped_indices.setdefault(day_key, []).append(idx)

    total_llm_count = 0
    total_cache_count = 0

    for day_key in sorted(grouped_indices.keys()):
        day_indices = grouped_indices[day_key]
        total_day_count = len(day_indices)
        day_llm_count = 0
        day_cache_count = 0

        # Batch count: e.g. 99 items, size=10 → ceil(99/10) = 10 batches
        batch_count = (total_day_count + effective_batch_size - 1) // effective_batch_size
        last_batch_size = total_day_count - effective_batch_size * (batch_count - 1)
        invalid_count = invalid_indices.get(day_key, 0)

        print(
            f"[Agent1] 📦 [{day_key}] {total_day_count} valid items → "
            f"{batch_count} batches (size={effective_batch_size}, last={last_batch_size}), "
            f"invalid={invalid_count}"
        )

        # -------------------------------------------------------------- #
        # Iterate in steps of effective_batch_size                        #
        # 99 items → batch_start: 0, 10, 20, ..., 90                     #
        # Last batch: indices[90:99] → 9 items                           #
        # -------------------------------------------------------------- #
        for batch_start in range(0, total_day_count, effective_batch_size):
            batch_indices = day_indices[batch_start : batch_start + effective_batch_size]
            batch_texts = [texts[k] for k in batch_indices]
            actual_batch_size = len(batch_indices)  # Naturally ≤ effective_batch_size

            batch_num = batch_start // effective_batch_size + 1
            print(
                f"[Agent1] 🚀 [{day_key}] Batch {batch_num}/{batch_count} "
                f"(items {batch_start + 1}–{batch_start + actual_batch_size}, "
                f"size={actual_batch_size}, day_total={total_day_count})"
            )

            # ---------------------------------------------------------- #
            # Check cache first                                           #
            # ---------------------------------------------------------- #
            uncached_texts: list[str] = []
            uncached_original_indices: list[int] = []

            for original_idx, text in zip(batch_indices, batch_texts):
                key = _sentiment_cache_key(day_key, text)
                cached_value = sentiment_cache.get(key)
                if (
                    not refresh_existing
                    and isinstance(cached_value, dict)
                    and "label" in cached_value
                    and "score" in cached_value
                ):
                    results[original_idx] = {
                        "label": str(cached_value.get("label", "neutral_info")),
                        "score": float(cached_value.get("score", 0.0)),
                        "confidence": float(cached_value.get("confidence", 0.0)),
                        "reason": str(cached_value.get("reason", "")),
                    }
                    day_cache_count += 1
                else:
                    uncached_original_indices.append(original_idx)
                    uncached_texts.append(text)

            if not uncached_texts:
                print(f"[Agent1] ♻️  [{day_key}] Batch {batch_num} fully served from cache")
                continue

            day_llm_count += len(uncached_texts)
            total_llm_count += len(uncached_texts)

            # ---------------------------------------------------------- #
            # Build prompt                                                #
            # ---------------------------------------------------------- #
            prompt = (
                "You are a senior crypto market psychology and discourse analyst. "
                "Your task is to perform nuanced sentiment and stance analysis for each text. "
                "Avoid simple positive/negative/neutral. Use the fine-grained labels below and "
                "reflect both intensity and confidence.\n\n"
                "Allowed labels (choose exactly one):\n"
                "- euphoric_bullish: extreme FOMO, hype, aggressive upside claims\n"
                "- strong_bullish: clear bullish stance with high conviction\n"
                "- optimistic: mildly bullish, measured tone\n"
                "- cautious_optimistic: bullish but highlights risks/conditions\n"
                "- neutral_info: factual or news-like, minimal emotion\n"
                "- mixed_uncertain: mixed signals, waiting for clarity\n"
                "- cautious_bearish: bearish but measured, risk-focused\n"
                "- fearful: anxiety or fear of continued downside\n"
                "- capitulation: despair, panic, \"dump/going to zero\" style\n"
                "- sarcastic_ironic: sarcastic tone; infer true stance in reason\n\n"
                "Output fields for each item:\n"
                "- sentiment_score: float in [-1, 1] (1 = euphoric, 0 = neutral, -1 = panic)\n"
                "- label: one of the allowed labels\n"
                "- confidence: float in [0, 1] for judgment certainty\n"
                "- reason: short English reason citing cues/phrases or context\n\n"
                "Rules:\n"
                "1) Distinguish factual reporting vs opinions.\n"
                "2) Consider emojis, hedges, conditions (e.g., \"if it breaks...\").\n"
                "3) If sarcastic, keep label sarcastic_ironic and explain the true lean.\n"
                "4) Keep each reason under 20 words to avoid JSON truncation.\n\n"
                "Respond ONLY with a valid JSON array (no markdown, no extra text) "
                "in the EXACT same order as the input texts.\n"
                "Format: [{\"sentiment_score\": <float>, \"label\": \"<category>\", "
                "\"confidence\": <float>, \"reason\": \"<short reason>\"}, ...]\n\n"
            )
            for idx, text in enumerate(uncached_texts):
                safe_text = text.replace("\n", " ").replace('"', "'")[:400]
                prompt += f"Text {idx}: {safe_text}\n"

            # ---------------------------------------------------------- #
            # Call LLM with retry on JSON parse failure                  #
            # ---------------------------------------------------------- #
            max_retries = 2
            parsed_data = None

            for attempt in range(max_retries + 1):
                response_text = ask_llm(prompt)

                if response_text == "ERROR":
                    print(
                        f"  -> ⚠️ API failure on batch {batch_num} "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"{'Retrying...' if attempt < max_retries else 'Giving up, using neutral fallback.'}"
                    )
                    if attempt < max_retries:
                        time.sleep(3)
                    continue

                try:
                    parsed_data = _parse_llm_json(response_text)
                    if parsed_data is not None:
                        break  # ✅ Success
                    else:
                        print(
                            f"  -> ⚠️ Output was not a JSON array on batch {batch_num} "
                            f"(attempt {attempt + 1}/{max_retries + 1}). "
                            f"{'Retrying...' if attempt < max_retries else 'Using neutral fallback.'}"
                        )
                except json.JSONDecodeError as e:
                    print(
                        f"  -> ⚠️ JSON parse error on batch {batch_num} "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"{'Retrying in 3s...' if attempt < max_retries else 'Using neutral fallback.'}"
                    )

                if attempt < max_retries:
                    time.sleep(3)

            # ---------------------------------------------------------- #
            # Map results back to original indices                        #
            # ---------------------------------------------------------- #
            if parsed_data is not None:
                for j, item in enumerate(parsed_data):
                    if j >= len(uncached_original_indices):
                        break
                    original_idx = uncached_original_indices[j]
                    score = float(item.get("sentiment_score", 0.0))
                    label = str(item.get("label", "neutral_info"))
                    confidence = float(item.get("confidence", 0.0))
                    reason = str(item.get("reason", ""))
                    normalized = {
                        "label": label,
                        "score": round(score, 4),
                        "confidence": round(confidence, 4),
                        "reason": reason,
                    }
                    results[original_idx] = normalized
                    cache_key = _sentiment_cache_key(day_key, texts[original_idx])
                    sentiment_cache[cache_key] = normalized
                    cache_updated = True
            else:
                print(f"  -> ⚠️ Batch {batch_num} exhausted retries. Neutral fallback applied.")
                for original_idx in uncached_original_indices:
                    normalized = {
                        "label": "neutral_info",
                        "score": 0.0,
                        "confidence": 0.0,
                        "reason": "fallback_after_llm_failure",
                    }
                    results[original_idx] = normalized
                    cache_key = _sentiment_cache_key(day_key, texts[original_idx])
                    sentiment_cache[cache_key] = normalized
                    cache_updated = True

            # Checkpoint-save cache after every batch
            if use_cache and cache_updated:
                _save_sentiment_cache(cache_path, sentiment_cache)
                cache_updated = False
                print(f"[Agent1] 💾 Cache checkpoint saved after batch {batch_num}: {cache_path}")

            # Wait briefly between batches to reduce API truncation risk.
            time.sleep(2)

        total_cache_count += day_cache_count
        print(
            f"[Agent1] ✅ Day {day_key} completed: "
            f"total={total_day_count}, llm={day_llm_count}, cache={day_cache_count}"
        )
        if callable(on_day_complete):
            on_day_complete(day_key, day_indices, results)

    # Final cache flush
    if use_cache:
        _save_sentiment_cache(cache_path, sentiment_cache)
        print(f"[Agent1] 💾 Sentiment cache flushed: {cache_path}")

    print(
        f"[Agent1] 📊 Total processed: {len(valid_indices)} items, "
        f"llm={total_llm_count}, cache={total_cache_count}"
    )
    return results


def fetch_tweets_from_csv(
    csv_path: str | list[str] = "bitcoin_tweets_cleaned.csv",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Load tweet data from one or more local CSVs and optionally filter by date range."""
    if isinstance(csv_path, str):
        csv_paths = [p.strip() for p in csv_path.split(",") if p.strip()]
    else:
        csv_paths = [str(p).strip() for p in csv_path if str(p).strip()]

    dataframes = []
    for path_str in csv_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            print(f"[Agent1] Tweet CSV not found: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[Agent1] Failed to read tweet CSV {path.name}: {e}")
            continue

        if "date" not in df.columns or "text" not in df.columns:
            print(f"[Agent1] Tweet CSV missing required columns: date/text ({path.name})")
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        if df.empty:
            continue

        df["_source_file"] = path.name
        dataframes.append(df)

    if not dataframes:
        print("[Agent1] No usable tweet CSV data found")
        return []

    df = pd.concat(dataframes, ignore_index=True)
    dedupe_cols = [c for c in ["date", "text", "user_name"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    if start_date:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_ts):
            df = df[df["date"] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_ts):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df["date"] <= end_ts]

    if df.empty:
        print("[Agent1] No tweets found in requested date range")
        return []

    items = []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        user_name = str(row.get("user_name", "") or "").strip()
        # Normalize to UTC-like ISO format for stable day grouping.
        dt = row["date"]
        if dt.tzinfo is None:
            published_at = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        else:
            published_at = dt.isoformat()
        items.append(
            {
                "source": "twitter_csv",
                "title": user_name,
                "text": text,
                "published_at": published_at,
            }
        )

    print(f"[Agent1] Loaded {len(items)} tweets from local CSV: {path.name}")
    return items


def fetch_news_from_csv(
    csv_path: str = "historical_news.csv",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Load news data from local CSV and optionally filter by date range."""
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        print(f"[Agent1] News CSV not found: {path}")
        return []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Agent1] Failed to read news CSV: {e}")
        return []

    if "date" not in df.columns or "title" not in df.columns:
        print("[Agent1] News CSV missing required columns: date/title")
        return []

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    if start_date:
        start_ts = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_ts):
            df = df[df["date"] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_ts):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df["date"] <= end_ts]

    if df.empty:
        print("[Agent1] No news found in requested date range")
        return []

    items = []
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "").strip()
        description = str(row.get("description", "") or "").strip()
        text = f"{title}. {description}".strip()
        if not text:
            continue

        source = str(row.get("source", "historical_news_csv")).strip()
        # Normalize to UTC-like ISO format for stable day grouping.
        raw_published = str(row.get("publishedAt", "")).strip()
        if raw_published:
            published_at = raw_published
        else:
            dt = row["date"]
            if dt.tzinfo is None:
                published_at = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            else:
                published_at = dt.isoformat()

        items.append(
            {
                "source": "news_csv",
                "title": title[:50] + "..." if len(title) > 50 else title,
                "text": text,
                "published_at": published_at,
            }
        )

    print(f"[Agent1] Loaded {len(items)} news articles from local CSV: {path.name}")
    return items


def _merge_items_by_day(
    tweet_items: list[dict],
    news_items: list[dict],
    per_day_limit: int = 0,
) -> list[dict]:
    """Merge tweets and news by day. If per_day_limit <= 0, include everything."""

    def _day_key(item: dict) -> str:
        published_at = str(item.get("published_at", "") or "")
        ts = pd.to_datetime(published_at, errors="coerce", utc=True)
        return str(ts.date()) if pd.notna(ts) else "unknown_day"

    grouped_tweets: dict[str, list[dict]] = {}
    for item in tweet_items:
        grouped_tweets.setdefault(_day_key(item), []).append(item)

    grouped_news: dict[str, list[dict]] = {}
    for item in news_items:
        grouped_news.setdefault(_day_key(item), []).append(item)

    all_days = sorted(set(grouped_tweets.keys()) | set(grouped_news.keys()))
    merged: list[dict] = []

    for day in all_days:
        day_tweets = grouped_tweets.get(day, [])
        day_news = grouped_news.get(day, [])
        if not day_tweets and not day_news:
            continue

        if per_day_limit > 0:
            if day_tweets and day_news:
                half = per_day_limit // 2
                chosen_tweets = day_tweets[:half]
                chosen_news = day_news[:half]
                remaining = per_day_limit - (len(chosen_tweets) + len(chosen_news))
                if remaining > 0:
                    extra_tweets = day_tweets[len(chosen_tweets):]
                    extra_news = day_news[len(chosen_news):]
                    extras = extra_tweets + extra_news
                    merged.extend(chosen_tweets + chosen_news + extras[:remaining])
                    selected_count = len(chosen_tweets) + len(chosen_news) + min(remaining, len(extras))
                else:
                    merged.extend(chosen_tweets + chosen_news)
                    selected_count = len(chosen_tweets) + len(chosen_news)
            else:
                source_items = day_tweets if day_tweets else day_news
                merged.extend(source_items[:per_day_limit])
                selected_count = min(per_day_limit, len(source_items))
        else:
            merged.extend(day_tweets + day_news)
            selected_count = len(day_tweets) + len(day_news)

        print(
            f"[Agent1] 📅 [{day}] tweets={len(day_tweets)}, news={len(day_news)}, selected={selected_count}"
        )

    return merged


def run_news_sentiment_agent(
    days: int = 7,
    csv_path: str = "bitcoin_tweets_cleaned.csv",
    news_csv_path: str = "historical_news.csv",
    start_date: str | None = None,
    end_date: str | None = None,
    per_day_batch_size: int = 10,
    use_cache: bool = True,
    cache_file: str = "artifacts/sentiment_cache.json",
    refresh_existing: bool = False,
) -> dict:
    """Run Agent 1 and return sentiment analysis results"""
    print("\n" + "=" * 60)
    print("🤖 Agent 1: News and Sentiment Analysis Starting...")
    print("=" * 60)

    if start_date is None:
        start_date = os.getenv("SENTIMENT_START_DATE")
    if end_date is None:
        end_date = os.getenv("SENTIMENT_END_DATE")
    tweets_env = os.getenv("SENTIMENT_TWEETS_CSV")
    if tweets_env:
        csv_path = tweets_env
    news_env = os.getenv("SENTIMENT_NEWS_CSV")
    if news_env:
        news_csv_path = news_env
    per_day_limit = int(os.getenv("SENTIMENT_PER_DAY_LIMIT", "0"))

    refresh_env = os.getenv("SENTIMENT_REFRESH_CACHE")
    if refresh_env is not None:
        refresh_existing = refresh_env.lower() in ("true", "1", "yes")

    cache_path = Path(cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    # Day-level cache hit: return directly when all requested days are cached.
    normalized_start = _normalize_day_key(start_date)
    normalized_end = _normalize_day_key(end_date)
    if use_cache and not refresh_existing and normalized_start and normalized_end:
        target_days = _iter_day_range(normalized_start, normalized_end)
        if target_days:
            cached_items, hit_days, miss_days = _load_cached_items_by_days(cache_path, target_days)
            if not miss_days and cached_items:
                print(
                    f"[Agent1] ♻️ Date-range cache hit: days={len(hit_days)} "
                    f"({hit_days[0]} -> {hit_days[-1]}), items={len(cached_items)}. Skip fetching."
                )
                scores = [float(item.get("sentiment_score", 0.0)) for item in cached_items]
                confidences = [float(item.get("sentiment_confidence", 0.0)) for item in cached_items]
                avg_score = float(np.mean(scores)) if scores else 0.0
                avg_confidence = float(np.mean(confidences)) if confidences else 0.0
                conf_sum = float(np.sum(confidences)) if confidences else 0.0
                weighted_score = (
                    float(np.sum(np.array(scores) * np.array(confidences))) / conf_sum
                    if conf_sum > 1e-12
                    else avg_score
                )
                overall_label = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"
                positive_count = int(sum(1 for s in scores if s > 0.1))
                negative_count = int(sum(1 for s in scores if s < -0.1))
                neutral_count = int(len(scores) - positive_count - negative_count)
                summary_text = (
                    f"Analyzed {len(cached_items)} cached articles. "
                    f"Overall sentiment is {overall_label} with an average score of {avg_score:.2f}, "
                    f"weighted score {weighted_score:.2f}, "
                    f"average confidence {avg_confidence:.2f}, "
                    f"positive={positive_count}, neutral={neutral_count}, negative={negative_count}."
                )
                return {
                    "avg_sentiment_score": avg_score,
                    "weighted_sentiment_score": weighted_score,
                    "avg_sentiment_confidence": avg_confidence,
                    "sentiment_label": overall_label,
                    "items_count": len(cached_items),
                    "positive_factors_count": positive_count,
                    "negative_factors_count": negative_count,
                    "neutral_factors_count": neutral_count,
                    "items": cached_items,
                    "scores_series": scores,
                    "gpt_summary": summary_text,
                }
            if miss_days:
                print(
                    f"[Agent1] 🧩 Date-range cache partial hit: hit={len(hit_days)}, miss={len(miss_days)}. "
                    f"Continue fetching for full rebuild."
                )

    tweet_items = fetch_tweets_from_csv(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
    )

    news_items = fetch_news_from_csv(
        csv_path=news_csv_path,
        start_date=start_date,
        end_date=end_date,
    )
    all_items = _merge_items_by_day(tweet_items, news_items, per_day_limit=per_day_limit)

    if not all_items:
        news = fetch_news(days=days)
        all_items = news

    if not all_items:
        print("[Agent1] Warning: No data fetched, using default neutral sentiment")
        return {
            "avg_sentiment_score": 0.0,
            "weighted_sentiment_score": 0.0,
            "avg_sentiment_confidence": 0.0,
            "sentiment_label": "neutral",
            "items_count": 0,
            "positive_factors_count": 0,
            "negative_factors_count": 0,
            "neutral_factors_count": 0,
            "items": [],
        }

    total_news_count = len(news_items)
    total_tweets_count = len(tweet_items)
    total_items_count = len(all_items)
    print(
        f"[Agent1] 📊 Input summary: {total_news_count} news + {total_tweets_count} tweets "
        f"= {total_items_count} total → sending to LLM"
    )

    texts = [item["text"] for item in all_items]
    published_ats = [str(item.get("published_at", "") or "") for item in all_items]

    def _checkpoint_day_results(day_key: str, day_indices: list[int], result_rows: list[dict]) -> None:
        if not use_cache:
            return
        day_items: list[dict] = []
        for idx in day_indices:
            item = dict(all_items[idx])
            sentiment = result_rows[idx]
            item["sentiment_score"] = sentiment["score"]
            item["sentiment_label"] = sentiment["label"]
            item["sentiment_confidence"] = sentiment.get("confidence", 0.0)
            item["sentiment_reason"] = sentiment.get("reason", "")
            day_items.append(item)
        _save_day_results_to_cache(cache_path, day_items)
        print(f"[Agent1] 💾 Day-level checkpoint saved: {day_key} ({len(day_items)} items)")

    sentiments = llm_sentiment_analysis(
        texts,
        published_ats=published_ats,
        per_day_batch_size=per_day_batch_size,
        use_cache=use_cache,
        cache_file=cache_file,
        refresh_existing=refresh_existing,
        on_day_complete=_checkpoint_day_results,
    )

    for i, item in enumerate(all_items):
        item["sentiment_score"] = sentiments[i]["score"]
        item["sentiment_label"] = sentiments[i]["label"]
        item["sentiment_confidence"] = sentiments[i].get("confidence", 0.0)
        item["sentiment_reason"] = sentiments[i].get("reason", "")

    scores = [s["score"] for s in sentiments]
    confidences = [float(s.get("confidence", 0.0)) for s in sentiments]
    avg_score = float(np.mean(scores))
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    conf_sum = float(np.sum(confidences)) if confidences else 0.0
    weighted_score = (
        float(np.sum(np.array(scores) * np.array(confidences))) / conf_sum
        if conf_sum > 1e-12
        else avg_score
    )
    overall_label = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"
    positive_count = int(sum(1 for s in scores if s > 0.1))
    negative_count = int(sum(1 for s in scores if s < -0.1))
    neutral_count = int(len(scores) - positive_count - negative_count)

    print(f"[Agent1] Average sentiment score: {avg_score:.4f} → {overall_label}")

    summary_text = (
        f"Analyzed {len(all_items)} recent articles. "
        f"Overall sentiment is {overall_label} with an average score of {avg_score:.2f}, "
        f"weighted score {weighted_score:.2f}, "
        f"average confidence {avg_confidence:.2f}, "
        f"positive={positive_count}, neutral={neutral_count}, negative={negative_count}."
    )

    if use_cache:
        _save_day_results_to_cache(cache_path, all_items)
        print(f"[Agent1] 💾 Day-level cache saved: {cache_path}")

    return {
        "avg_sentiment_score": avg_score,
        "weighted_sentiment_score": weighted_score,
        "avg_sentiment_confidence": avg_confidence,
        "sentiment_label": overall_label,
        "items_count": len(all_items),
        "positive_factors_count": positive_count,
        "negative_factors_count": negative_count,
        "neutral_factors_count": neutral_count,
        "items": all_items,
        "scores_series": scores,
        "gpt_summary": summary_text,
    }


if __name__ == "__main__":
    result = run_news_sentiment_agent(
        start_date="2025-03-10",
        end_date="2026-03-10",
        per_day_batch_size=10,  # Process 10 items per batch; remainder goes to the final batch.
        use_cache=True,         # Enable cache.
        refresh_existing=False   # Keep existing cache entries.
    )
    print("\nFinal sentiment score:", result["avg_sentiment_score"])
