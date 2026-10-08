"""Telegram Bot Interface for Space RAG System with Memory & Clean HTML formatting."""

import asyncio
import logging
import os
import re
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import Command
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(LAB_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

groq_api_key = os.getenv("GROQ_API_KEY")
bot_token = os.getenv("BOT_TOKEN")

cfg = yaml.safe_load((LAB_ROOT / "configs" / "rag.yaml").read_text(encoding="utf-8"))

db_dir = LAB_ROOT / cfg["db_path"]
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=cfg["embedding_model"]
)
chroma_client = chromadb.PersistentClient(path=str(db_dir))
collection = chroma_client.get_collection(
    name=cfg["collection_name"], embedding_function=embed_fn
)

groq_client = Groq(api_key=groq_api_key)
bot = Bot(token=bot_token)
dp = Dispatcher()

# Хранилище истории диалогов по chat_id
user_history = {}


def retrieve_context(query: str, top_k: int = 5):
    """Поиск наилучших чанков в ChromaDB."""
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


def clean_html_for_telegram(text: str) -> str:
    """Очистка ответа от лишних HTML-тегов, которые не поддерживаются Telegram API."""
    # Удаляем ```html и ``` в начале/конце
    text = re.sub(r"^```html\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    
    # Удаляем теги html, body, head, doctype
    text = re.sub(r"</?(?:html|body|head|doctype)[^>]*>", "", text, flags=re.IGNORECASE)
    
    return text.strip()


def generate_answer(query: str, retrieved_docs: list, history: list) -> str:
    """Генерация ответа через Groq с учетом контекста и истории."""
    sources_list = "\n".join([f"• <a href='{d['url']}'>{d['title']}</a>" for d in retrieved_docs])
    context_str = "\n\n".join(
        [f"--- Источник: {d['title']} ---\n{d['text']}" for d in retrieved_docs]
    )

    system_prompt = (
        "Вы — ассистент-эксперт по теме 'Освоение космоса'. "
        "Отвечайте на вопрос пользователя ТОЛЬКО на основе предоставленного контекста и истории беседы. "
        "Используйте только поддерживаемые Telegram теги HTML: <b>текст</b> для выделения и <i>текст</i> для акцента. "
        "НЕ ИСПОЛЬЗУЙТЕ теги <html>, <body>, <p>, <div>, <h1>, <h2>. Пишите обычным текстом с абзацами."
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    # История переписки
    for h in history[-4:]:
        messages.append(h)

    user_content = f"Контекст из Википедии:\n{context_str}\n\nВопрос пользователя: {query}"
    messages.append({"role": "user", "content": user_content})

    response = groq_client.chat.completions.create(
        model=cfg["groq_model"],
        messages=messages,
        temperature=0.2,
    )

    raw_answer = response.choices[0].message.content
    cleaned_answer = clean_html_for_telegram(raw_answer)

    # Добавляем кликабельные источники
    final_output = f"{cleaned_answer}\n\n<b>Использованные источники:</b>\n{sources_list}"
    return final_output


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_history[message.chat.id] = []
    await message.answer(
        "🚀 <b>Привет! Я RAG-ассистент по освоению космоса.</b>\n\n"
        "Задайте мне любой вопрос про космические миссии, аппараты или космонавтов!",
        parse_mode=ParseMode.HTML
    )


@dp.message()
async def handle_question(message: types.Message):
    chat_id = message.chat.id
    user_query = message.text.strip()
    if not user_query:
        return

    if chat_id not in user_history:
        user_history[chat_id] = []

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        search_query = user_query
        if len(user_history[chat_id]) > 0 and len(user_query.split()) < 5:
            last_user_msg = user_history[chat_id][-2]["content"] if len(user_history[chat_id]) >= 2 else ""
            search_query = f"{last_user_msg} {user_query}"

        retrieved = retrieve_context(search_query, top_k=cfg.get("top_k", 5))
        answer = generate_answer(user_query, retrieved, user_history[chat_id])

        user_history[chat_id].append({"role": "user", "content": user_query})
        user_history[chat_id].append({"role": "assistant", "content": answer})

        await message.answer(answer, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    except Exception as exc:
        log.error("Ошибка при обработке запроса: %s", exc)
        # Если Telegram всё равно не смог спарсить HTML, отправляем без парсинга
        try:
            plain_answer = re.sub(r"<[^>]+>", "", answer)
            await message.answer(plain_answer)
        except Exception:
            await message.answer("⚠️ Произошла ошибка при форматировании ответа.")


async def main():
    log.info("Запуск Telegram-бота...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())