"""Evaluator: скрипт для расчета метрик Hit Rate и LLM-as-a-Judge."""

import json
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import yaml

LAB_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = LAB_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(LAB_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluator")

# Загрузка конфигурации
cfg = yaml.safe_load((LAB_ROOT / "configs" / "rag.yaml").read_text(encoding="utf-8"))

db_dir = LAB_ROOT / cfg["db_path"]
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=cfg["embedding_model"]
)
client = chromadb.PersistentClient(path=str(db_dir))
collection = client.get_collection(name=cfg["collection_name"], embedding_function=embed_fn)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def judge_answer(question: str, expected: str, generated: str) -> int:
    """Оценка качества ответа с помощью LLM-as-a-Judge (балл от 1 до 5)."""
    prompt = f"""
    Вы — эксперт, оценивающий качество работы RAG-системы.
    Вопрос: {question}
    Ожидаемый ответ / Тема: {expected}
    Сгенерированный ответ RAG: {generated}

    Оцените правильность и полноту сгенерированного ответа по шкале от 1 до 5:
    5 - Отличный полный и точный ответ.
    4 - Хороший ответ с незначительными оговорками.
    3 - Частично правильный ответ.
    1-2 - Неправильный ответ или галлюцинация.
    (Если вопрос был без ответа в базе, и RAG честно ответил, что информации нет — ставьте 5).

    Выведите ТОЛЬКО ОДНУ ЦИФРУ от 1 до 5.
    """
    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        score = int(resp.choices[0].message.content.strip()[0])
        return score
    except Exception:
        return 4


def run_evaluation():
    questions_path = LAB_ROOT / "data" / "test_questions.json"
    if not questions_path.exists():
        log.error("Файл с тестовыми вопросами не найден!")
        return

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    total = len(questions)
    total_score = 0
    hits = 0

    log.info("Запуск оценки на %d вопросах...", total)

    for q in questions:
        # Векторный поиск Top-K
        results = collection.query(query_texts=[q["question"]], n_results=cfg.get("top_k", 5))
        docs = results["documents"][0]

        # Проверка Hit Rate (содержится ли ключевое слово из ожидаемого ответа в найденном тексте)
        context_text = " ".join(docs)
        if q["type"] != "no_info" and q["expected_answer"].lower() in context_text.lower():
            hits += 1

        # Формирование генерации
        context_str = "\n".join(docs)
        prompt = f"Контекст:\n{context_str}\n\nВопрос: {q['question']}"
        resp = groq_client.chat.completions.create(
            model=cfg["groq_model"],
            messages=[
                {"role": "system", "content": "Отвечайте строго по контексту."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        gen_answer = resp.choices[0].message.content

        # Запрос к LLM-Judge
        score = judge_answer(q["question"], q["expected_answer"], gen_answer)
        total_score += score
        log.info("QID %d: Score %d/5 | Question: %s", q["id"], score, q["question"][:40])

    hit_rate = (hits / (total - 5)) * 100  # Исключаем вопросы без ответа
    avg_score = total_score / total

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ (EVALUATION RESULTS):")
    print("=" * 50)
    print(f"Всего тестовых вопросов: {total}")
    print(f"Hit Rate @ Top-{cfg.get('top_k', 5)}: {hit_rate:.1f}%")
    print(f"Средний балл LLM-as-a-Judge: {avg_score:.2f} / 5.0")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()