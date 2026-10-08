"""Grabber: собирает статьи русской Википедии по поисковым запросам."""
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]  # папка lab1
API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "pad-lab-rag/0.1 (student lab project; mailto:your_email@example.com)"
}
log = logging.getLogger("grabber")


def api_get(session, params, retries=5):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait_time = 5 * attempt
                log.warning("Превышен лимит запросов (429). Ждём %d сек...", wait_time)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Ошибка запроса (попытка %d/%d): %s", attempt, retries, exc)
            time.sleep(2 ** attempt)
    return None

def search_titles(session, query, limit):
    data = api_get(session, {
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": limit, "srnamespace": 0,
    })
    if not data:
        return []
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def fetch_page(session, title):
    data = api_get(session, {
        "action": "query", "prop": "extracts|info", "explaintext": 1,
        "exsectionformat": "wiki", "inprop": "url",
        "redirects": 1, "titles": title,
    })
    if not data:
        return None
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    p = pages[0]
    return {
        "page_id": p["pageid"],
        "title": p["title"],
        "url": p["fullurl"],
        "revision_id": p["lastrevid"],
        "text": p.get("extract", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "grabber.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    session = requests.Session()
    delay = cfg["delay_sec"]

    titles = []
    for query in cfg["queries"]:
        found = search_titles(session, query, cfg["results_per_query"])
        log.info("Запрос %r: найдено %d", query, len(found))
        titles.extend(t for t in found if t not in titles)
        time.sleep(delay)
    titles = titles[: cfg["max_docs"]]

    stats = {"new": 0, "updated": 0, "unchanged": 0, "skipped": 0, "failed": 0}
    try:
        for title in titles:
            page = fetch_page(session, title)
            time.sleep(delay)
            if page is None:
                stats["failed"] += 1
                continue
            if len(page["text"]) < cfg["min_chars"]:
                stats["skipped"] += 1
                continue
            key = str(page["page_id"])
            old = manifest.get(key)
            if old and old["revision_id"] == page["revision_id"]:
                stats["unchanged"] += 1
                continue
            page["content_hash"] = hashlib.sha256(page["text"].encode("utf-8")).hexdigest()
            page["fetched_at"] = datetime.now(timezone.utc).isoformat()
            (raw_dir / f"{key}.json").write_text(
                json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest[key] = {k: page[k] for k in
                             ("title", "url", "revision_id", "content_hash", "fetched_at")}
            stats["updated" if old else "new"] += 1
            log.info("Сохранено: %s", page["title"])
    finally:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Итого: %s", stats)


if __name__ == "__main__":
    main()