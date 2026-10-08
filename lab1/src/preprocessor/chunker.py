"""Preprocessor & Chunker: очищает текст и режет статьи на смысловые чанки."""

import argparse
import json
import logging
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]  # Папка lab1
log = logging.getLogger("preprocessor")


def clean_text(text: str) -> str:
    """Очистка текста от лишних пустых строк."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int, min_chunk_size: int):
    """Стратегия нарезки по абзацам с перекрытием (overlap)."""
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += ("\n" if current_chunk else "") + para
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            tail = current_chunk[overlap_start:]
            current_chunk = tail + ("\n" if tail else "") + para

    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "preprocessor.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(raw_dir.glob("*.json"))
    log.info("Найдено сырых файлов: %d", len(raw_files))

    all_chunks = []
    total_articles = 0

    for raw_file in raw_files:
        article = json.loads(raw_file.read_text(encoding="utf-8"))
        cleaned = clean_text(article.get("text", ""))

        text_chunks = split_text_into_chunks(
            cleaned,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            min_chunk_size=cfg["min_chunk_size"],
        )

        for idx, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{article['page_id']}_{idx}",
                "page_id": article["page_id"],
                "title": article["title"],
                "url": article["url"],
                "chunk_index": idx,
                "text": chunk_text,
            })
        total_articles += 1

    chunks_file = processed_dir / "chunks.json"
    chunks_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Обработано статей: %d. Всего создано чанков: %d", total_articles, len(all_chunks))
    log.info("Сохранено в: %s", chunks_file)


if __name__ == "__main__":
    main()