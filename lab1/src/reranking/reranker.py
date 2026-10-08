"""Reranker: переранжирует полученные чанки с помощью модели FlashRank."""

from flashrank import Ranker, RerankRequest


class SpaceReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, retrieved_docs: list, top_n: int = 5) -> list:
        """
        Принимает список чанков, пересортирует их и возвращает top_n лучших.
        """
        if not retrieved_docs:
            return []

        passages = [
            {"id": idx, "text": doc["text"], "meta": doc}
            for idx, doc in enumerate(retrieved_docs)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_req)

        # Отбираем Top-N переранжированных результатов
        reranked_docs = [item["meta"] for item in results[:top_n]]
        return reranked_docs