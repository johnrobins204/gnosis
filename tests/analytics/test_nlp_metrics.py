import pytest
from src.analytics.metrics.nlp import BLEUMetric, ROUGEMetric, BERTScoreMetric

def test_bleu_metric():
    metric = BLEUMetric()
    data = {"references": ["This is a test"], "hypotheses": ["This is a test"]}
    result = metric.calculate(data)
    assert result[0] > 0.9  # BLEU score should be high for identical texts

def test_rouge_metric():
    metric = ROUGEMetric()
    data = {"references": ["This is a test"], "hypotheses": ["This is a test"]}
    result = metric.calculate(data)
    assert "rouge1" in result[0]

def test_bert_score_metric():
    metric = BERTScoreMetric()
    data = {"references": ["This is a test"], "hypotheses": ["This is a test"]}
    result = metric.calculate(data)
    assert "f1" in result