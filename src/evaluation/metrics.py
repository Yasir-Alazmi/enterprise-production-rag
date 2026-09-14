"""RAG evaluation harness implementing precision, recall, and grounding metrics."""

import re
from typing import Dict, List, Optional, Set


class RAGEvaluator:
    """Computes empirical quality metrics on retrieved context and generated answers."""

    @staticmethod
    def context_precision(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Calculate Context Precision@K based on ground truth relevance ranks."""
        if not retrieved_ids or not ground_truth_ids:
            return 0.0

        gt_set: Set[str] = set(ground_truth_ids)
        hits = 0
        precision_sum = 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gt_set:
                hits += 1
                precision_sum += hits / rank

        return round(precision_sum / max(1, min(len(retrieved_ids), len(ground_truth_ids))), 4)

    @staticmethod
    def context_recall(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Calculate Context Recall: fraction of ground truth documents retrieved."""
        if not ground_truth_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        hits = sum(1 for gt in ground_truth_ids if gt in retrieved_set)
        return round(hits / len(ground_truth_ids), 4)

    @staticmethod
    def faithfulness(answer: str, context_chunks: List[str]) -> float:
        """Measure what fraction of answer sentences are grounded in the retrieved context."""
        if not answer.strip() or not context_chunks:
            return 0.0

        # Strip bracket citations before splitting sentences to avoid fragmentation
        cleaned_answer = re.sub(r"\[.*?\]", "", answer).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        context_text = " ".join(context_chunks).lower()
        context_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context_text))

        supported = 0
        for sent in sentences:
            cleaned = sent.lower()
            for b in [
                "based on verified enterprise policies,",
                "furthermore,",
                "additionally,",
                "according to enterprise policy,"
            ]:
                if cleaned.startswith(b):
                    cleaned = cleaned[len(b):].strip()

            sent_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", cleaned))
            if not sent_words:
                continue
            overlap = len(sent_words.intersection(context_words)) / len(sent_words)
            if overlap >= 0.40:
                supported += 1

        return round(supported / max(1, len(sentences)), 4)

    @staticmethod
    def citation_correctness(answer: str, retrieved_citations: List[str]) -> float:
        """Verify that citations present in the answer correspond to valid retrieved sections."""
        citations_found = re.findall(r"\[(.*?)\]", answer)
        if not citations_found:
            return 1.0

        matches = 0
        for c in citations_found:
            c_clean = c.strip().lower()
            if any(c_clean in s.lower() or s.lower() in c_clean for s in retrieved_citations):
                matches += 1
        return round(matches / len(citations_found), 4)

    @staticmethod
    def mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for the first relevant item."""
        if not retrieved_ids or not ground_truth_ids:
            return 0.0

        gt_set: Set[str] = set(ground_truth_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gt_set:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
        if not retrieved_ids or not ground_truth_ids or k <= 0:
            return 0.0

        import math

        gt_set: Set[str] = set(ground_truth_ids)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in gt_set:
                dcg += 1.0 / math.log2(rank + 1)

        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(ground_truth_ids)) + 1))
        if idcg <= 0.0:
            return 0.0
        return round(dcg / idcg, 4)

    @staticmethod
    def answer_relevance(query: str, answer: str) -> float:
        """Compute keyword relevance overlap between query tokens and generated answer."""
        if not query.strip() or not answer.strip():
            return 0.0

        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower()))
        a_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", answer.lower()))
        if not q_tokens or not a_tokens:
            return 0.0

        overlap = len(q_tokens.intersection(a_tokens))
        return round(overlap / len(q_tokens), 4)

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        answer: str,
        context_chunks: List[str],
        query: str = "",
        retrieved_citations: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive empirical evaluation metrics for a query execution."""
        results = {
            "context_precision": self.context_precision(retrieved_ids, ground_truth_ids),
            "context_recall": self.context_recall(retrieved_ids, ground_truth_ids),
            "mrr": self.mrr(retrieved_ids, ground_truth_ids),
            "ndcg_at_5": self.ndcg_at_k(retrieved_ids, ground_truth_ids, k=5),
            "faithfulness": self.faithfulness(answer, context_chunks)
        }
        if query:
            results["answer_relevance"] = self.answer_relevance(query, answer)
        if retrieved_citations is not None:
            results["citation_correctness"] = self.citation_correctness(answer, retrieved_citations)
        return results
