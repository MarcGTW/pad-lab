"""RAG Pipeline: ретривер контекста из ChromaDB + генерация ответа через Groq API."""

import argparse
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import yaml

ROOT = Path(__file__).resolve().parents[1]  # папка lab1
load_dotenv(ROOT.parent / ".env")  # Загрузка GROQ_API_KEY из .env


def setup_chroma(cfg):
    """Инициализация подключения к ChromaDB."""
    db_dir = ROOT / cfg["db_path"]
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding_model"]
    )
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_collection(name=cfg["collection_name"], embedding_function=embed_fn)


def retrieve_context(collection, query: str, top_k: int):
    """Поиск наилучших контекстов для вопроса."""
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    retrieved = []
    for doc, meta in zip(docs, metas):
        retrieved.append({
            "text": doc,
            "title": meta.get("title", "Источник"),
            "url": meta.get("url", "#")
        })
    return retrieved


def generate_answer(query: str, retrieved_docs: list, model_name: str):
    """Формирование промпта и запрос к LLM через Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Ошибка: GROQ_API_KEY не найден в файле .env!")

    client = Groq(api_key=api_key)

    context_str = "\n\n".join(
        [f"--- Источник: {d['title']} ({d['url']}) ---\n{d['text']}" for d in retrieved_docs]
    )

    system_prompt = (
        "Вы — ассистент-эксперт по теме 'Освоение космоса'. "
        "Отвечайте на вопрос пользователя ТОЛЬКО на основе предоставленного контекста. "
        "Если в контексте нет прямого ответа, честно скажите об этом. "
        "В конце ответа приведите список использованных источников (названия и ссылки)."
    )

    user_prompt = f"Контекст:\n{context_str}\n\nВопрос: {query}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "rag.yaml"))
    parser.add_argument("--query", type=str, help="Вопрос к RAG системе")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    collection = setup_chroma(cfg)

    query = args.query
    if not query:
        query = input("Введите ваш вопрос про космос: ")

    print(f"\n[1/2] Поиск контекстов для: '{query}'...")
    retrieved = retrieve_context(collection, query, cfg["top_k"])

    print(f"[2/2] Генерация ответа через Groq ({cfg['groq_model']})...\n")
    answer = generate_answer(query, retrieved, cfg["groq_model"])

    print("=" * 60)
    print("ОТВЕТ RAG СИСТЕМЫ:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()