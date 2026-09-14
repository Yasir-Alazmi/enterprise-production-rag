"""Unit tests for MRR, NDCG, and answer relevance evaluation metrics."""

import pytest

from src.evaluation.metrics import RAGEvaluator


def test_mrr_first_rank():
    evaluator = RAGEvaluator()
    # Relevant document at rank 1 -> MRR = 1.0
    mrr = evaluator.mrr(retrieved_ids=["doc_1", "doc_2", "doc_3"], ground_truth_ids=["doc_1"])
    assert mrr == 1.0


def test_mrr_second_rank():
    evaluator = RAGEvaluator()
    # Relevant document at rank 2 -> MRR = 0.5
    mrr = evaluator.mrr(retrieved_ids=["doc_x", "doc_target", "doc_y"], ground_truth_ids=["doc_target"])
    assert mrr == 0.5


def test_mrr_no_match():
    evaluator = RAGEvaluator()
    mrr = evaluator.mrr(retrieved_ids=["doc_a", "doc_b"], ground_truth_ids=["doc_z"])
    assert mrr == 0.0


def test_ndcg_ideal():
    evaluator = RAGEvaluator()
    # Perfect ranking: both ground truth items at top positions
    ndcg = evaluator.ndcg_at_k(
        retrieved_ids=["gt_1", "gt_2", "noise_1"],
        ground_truth_ids=["gt_1", "gt_2"],
        k=3
    )
    assert pytest.approx(ndcg, abs=1e-3) == 1.0


def test_ndcg_imperfect():
    evaluator = RAGEvaluator()
    # Lower ranking gives lower NDCG than 1.0
    ndcg = evaluator.ndcg_at_k(
        retrieved_ids=["noise_1", "gt_1", "gt_2"],
        ground_truth_ids=["gt_1", "gt_2"],
        k=3
    )
    assert 0.0 < ndcg < 1.0


def test_answer_relevance():
    evaluator = RAGEvaluator()
    query = "What is the annual leave entitlement?"
    answer = "The annual leave entitlement for employees is thirty days per annum."
    score = evaluator.answer_relevance(query, answer)
    assert score > 0.40

    irrelevant = "Ransomware encryption occurred on Friday morning."
    score_low = evaluator.answer_relevance(query, irrelevant)
    assert score_low == 0.0
