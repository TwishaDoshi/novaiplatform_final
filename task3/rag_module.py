import os
import re
import json
from typing import List, Dict, Any

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


class NovaRAG:
    def __init__(self):
        self.products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
        self.returns_df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="nova_knowledge_base",
            embedding_function=self.embedding_fn
        )

        # lightweight reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def build_documents(self) -> List[Dict[str, Any]]:
        docs = []

        # Product docs
        for _, row in self.products.iterrows():
            product_id = str(row.get("product_id", ""))
            name = str(row.get("name", ""))
            category = str(row.get("category", ""))
            ingredients = str(row.get("ingredients", ""))
            target = str(row.get("target", ""))
            price = str(row.get("price", ""))

            text = (
                f"Product ID: {product_id}\n"
                f"Product Name: {name}\n"
                f"Category: {category}\n"
                f"Ingredients: {ingredients}\n"
                f"Suitable For: {target}\n"
                f"Price: {price}\n"
                f"Description: {name} is a {category} product. "
                f"It contains {ingredients} and is intended for {target}."
            )

            docs.append({
                "id": f"product_{product_id}",
                "text": text,
                "metadata": {
                    "source": "product_catalog",
                    "product_id": product_id,
                    "category": category,
                    "name": name
                }
            })

        # Policy docs
        policy_docs = [
            {
                "id": "policy_returns_1",
                "text": (
                    "NOVA return policy: Customers can request a return for eligible products "
                    "within 7 days of delivery. Damaged items can be returned with a valid reason. "
                    "Order ID is required to initiate a return."
                ),
                "metadata": {"source": "policy", "topic": "returns"}
            },
            {
                "id": "policy_shipping_1",
                "text": (
                    "NOVA shipping policy: Orders may be in processing, shipped, or delivered status. "
                    "Delivery dates are estimates and may vary depending on destination."
                ),
                "metadata": {"source": "policy", "topic": "shipping"}
            },
            {
                "id": "policy_sizing_1",
                "text": (
                    "NOVA sizing guidance: Customers should compare their usual fit with available "
                    "sizes such as S, M, L, and XL. Apparel fit guidance may vary by product."
                ),
                "metadata": {"source": "policy", "topic": "sizing"}
            },
            {
                "id": "policy_safety_1",
                "text": (
                    "NOVA safety guidance: If a customer reports allergy, skin irritation, rash, "
                    "or any adverse reaction, the issue should be escalated to a human support specialist."
                ),
                "metadata": {"source": "policy", "topic": "safety"}
            }
        ]
        docs.extend(policy_docs)

        # FAQ docs
        faq_docs = [
            {
                "id": "faq_ingredients_1",
                "text": (
                    "FAQ: Customers often ask whether skincare products are suitable for oily, dry, "
                    "or sensitive skin. Product suitability should be based on the product target field "
                    "and listed ingredients."
                ),
                "metadata": {"source": "faq", "topic": "ingredients"}
            },
            {
                "id": "faq_recommendation_1",
                "text": (
                    "FAQ: Product recommendations should consider skin type, category, and customer needs. "
                    "Hydrating products are often relevant for dry skin, while salicylic-acid-based cleansers "
                    "may be relevant for oily skin."
                ),
                "metadata": {"source": "faq", "topic": "recommendation"}
            },
            {
                "id": "faq_returns_1",
                "text": (
                    "FAQ: To process a return, support should ask for the order ID and the reason for return. "
                    "Damaged products can be prioritized for return handling."
                ),
                "metadata": {"source": "faq", "topic": "returns"}
            }
        ]
        docs.extend(faq_docs)

        return docs

    def ingest_documents(self):
        documents = self.build_documents()

        existing = self.collection.get()
        existing_ids = set(existing["ids"]) if existing and existing.get("ids") else set()

        new_ids = []
        new_texts = []
        new_metadatas = []

        for doc in documents:
            if doc["id"] not in existing_ids:
                new_ids.append(doc["id"])
                new_texts.append(doc["text"])
                new_metadatas.append(doc["metadata"])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metadatas
            )

        return {
            "ingested_count": len(new_ids),
            "total_known_docs": len(documents)
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs = []
        ids = results.get("ids", [[]])[0]
        texts = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, texts, metas, distances):
            docs.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "score": float(dist) if dist is not None else None
            })

        return docs

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not docs:
            return []

        pairs = [(query, d["text"]) for d in docs]
        scores = self.reranker.predict(pairs)

        reranked = []
        for doc, score in zip(docs, scores):
            updated = doc.copy()
            updated["rerank_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def answer_query(self, query: str, top_k: int = 5, rerank_k: int = 3) -> Dict[str, Any]:
        retrieved = self.retrieve(query=query, top_k=top_k)
        reranked = self.rerank(query=query, docs=retrieved, top_k=rerank_k)

        if not reranked:
            return {
                "query": query,
                "answer": "I could not find relevant information in the NOVA knowledge base.",
                "sources": []
            }

        # simple grounded synthesis
        top_texts = [doc["text"] for doc in reranked]
        answer = self._synthesize_answer(query, reranked)

        return {
            "query": query,
            "answer": answer,
            "sources": reranked
        }

    def _synthesize_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        q = query.lower()

        if "oily skin" in q:
            for d in docs:
                t = d["text"].lower()
                if "oily" in t:
                    return (
                        "Based on the knowledge base, this appears suitable for oily skin. "
                        "The answer is grounded in the product target and ingredient details retrieved from the catalog."
                    )

        if "dry skin" in q:
            for d in docs:
                t = d["text"].lower()
                if "dry" in t:
                    return (
                        "Based on the knowledge base, hydrating products targeted for dry skin are the most relevant match."
                    )

        if "return" in q or "refund" in q:
            return (
                "According to NOVA policy, returns can be requested for eligible products within 7 days of delivery, "
                "and the order ID is required to initiate the process."
            )

        if "size" in q or "fit" in q:
            return (
                "According to NOVA sizing guidance, customers should compare their usual fit with available sizes "
                "like S, M, L, and XL because fit can vary by apparel product."
            )

        # fallback: summarize first source
        top = docs[0]
        return f"Based on the retrieved knowledge, the most relevant information is: {top['text'][:240]}..."

    def evaluate(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        results = []
        correct = 0

        for case in test_cases:
            query = case["query"]
            expected_keyword = case["expected_keyword"].lower()

            result = self.answer_query(query)
            answer = result["answer"].lower()

            hit = expected_keyword in answer
            if hit:
                correct += 1

            results.append({
                "query": query,
                "expected_keyword": case["expected_keyword"],
                "answer": result["answer"],
                "hit": hit,
                "num_sources": len(result["sources"])
            })

        accuracy = correct / len(test_cases) if test_cases else 0.0

        return {
            "num_cases": len(test_cases),
            "accuracy": accuracy,
            "results": results
        }


if __name__ == "__main__":
    rag = NovaRAG()
    print(rag.ingest_documents())

    sample = rag.answer_query("Is this serum good for oily skin?")
    print(json.dumps(sample, indent=2))