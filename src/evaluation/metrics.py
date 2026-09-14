"""RAG evaluation harness implementing precision, recall, and grounding metrics."""

import re
from typing import Dict, List, Set

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

        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        context_text = " ".join(context_chunks).lower()
        context_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", context_text))

        supported = 0
        for sent in sentences:
            sent_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", sent.lower()))
            if not sent_words:
                continue
            overlap = len(sent_words.intersection(context_words)) / len(sent_words)
            if overlap >= 0.50:  # at least 50% keyword grounding
                supported += 1

        return round(supported / max(1, len(sentences)), 4)

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        answer: str,
        context_chunks: List[str]
    ) -> Dict[str, float]:
        """Compute end-to-end evaluation metrics for a query execution."""
        return {
            "context_precision": self.context_precision(retrieved_ids, ground_truth_ids),
            "context_recall": self.context_recall(retrieved_ids, ground_truth_ids),
            "faithfulness": self.faithfulness(answer, context_chunks)
        }
