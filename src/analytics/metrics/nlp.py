from src.analytics.metrics.base import Metric, MetricRegistry
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

class BLEUMetric(Metric):
    """BLEU metric for evaluating text similarity."""

    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25)):
        self.weights = weights

    def calculate(self, data):
        """Calculate BLEU score for a list of references and hypotheses."""
        if not data["references"] or not data["hypotheses"]:
            raise ValueError("References and hypotheses cannot be empty.")
        references, hypotheses = data["references"], data["hypotheses"]
        scores = [sentence_bleu([ref], hyp, weights=self.weights) for ref, hyp in zip(references, hypotheses)]
        return scores

    def name(self):
        return "BLEU"

class ROUGEMetric(Metric):
    """ROUGE metric for evaluating text similarity."""

    def __init__(self, rouge_types=["rouge1", "rouge2", "rougeL"]):
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def calculate(self, data):
        """Calculate ROUGE scores for a list of references and hypotheses."""
        references, hypotheses = data["references"], data["hypotheses"]
        scores = [self.scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        return scores

    def name(self):
        return "ROUGE"

class BERTScoreMetric(Metric):
    """BERTScore metric for semantic similarity."""

    def __init__(self, model_type="bert-base-uncased"):
        self.model_type = model_type

    def calculate(self, data):
        """Calculate BERTScore for a list of references and hypotheses."""
        references, hypotheses = data["references"], data["hypotheses"]
        P, R, F1 = score(hypotheses, references, model_type=self.model_type, verbose=False)
        return {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}

    def name(self):
        return "BERTScore"

# Register metrics
MetricRegistry.register(BLEUMetric)
MetricRegistry.register(ROUGEMetric)
MetricRegistry.register(BERTScoreMetric)