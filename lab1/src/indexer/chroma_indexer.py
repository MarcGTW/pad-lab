"""Indexer: генерирует эмбеддинги для чанков и сохраняет их в ChromaDB."""

import argparse
import json
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import yaml

ROOT = Path(__file__).resolve().parents[2]  # папка lab1
log = logging.getLogger("indexer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "indexer.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    chunks_file = ROOT / "data" / "processed" / "chunks.json"
    if not chunks_file.exists():
        log.error("Файл с чанками не найден: %s. Запустите сначала chunker.py!", chunks_file)
        return

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    log.info("Загружено чанков для индексации: %d", len(chunks))

    db_dir = ROOT / cfg["db_path"]
    db_dir.mkdir(parents=True, exist_ok=True)

    log.info("Инициализация модели эмбеддингов: %s", cfg["embedding_model"])
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding_model"]
    )

    client = chromadb.PersistentClient(path=str(db_dir))
    
    # Создаём или получаем коллекцию
    collection = client.get_or_create_collection(
        name=cfg["collection_name"],
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = cfg.get("batch_size", 256)
    total_chunks = len(chunks)

    log.info("Запись данных в ChromaDB пачками по %d...", batch_size)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]

        ids = [item["chunk_id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [
            {
                "page_id": str(item["page_id"]),
                "title": item["title"],
                "url": item["url"],
                "chunk_index": item["chunk_index"],
            }
            for item in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        log.info("Индексировано %d / %d чанков", min(i + batch_size, total_chunks), total_chunks)

    log.info("Индексация завершена! Всего элементов в коллекции '%s': %d", 
             cfg["collection_name"], collection.count())


if __name__ == "__main__":
    main()