from .base import Metric, MetricRegistry
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools

class SemanticDifferenceMetric(Metric):
    def __init__(self, response_col='raw_model_output'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.response_col = response_col

    def calculate(self, data):
        if self.response_col not in data:
            raise KeyError(f"Input data must have a '{self.response_col}' column or key.")
        responses = list(data[self.response_col])
        if len(responses) < 2:
            return 1.0
        embeddings = self.model.encode(responses, convert_to_numpy=True)
        pairs = itertools.combinations(range(len(embeddings)), 2)
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[j]])[0, 0]
            for i, j in pairs
        ]
        return float(np.mean(similarities))

    def name(self):
        return "semantic_difference"

# Register the metric with a custom name
MetricRegistry.register(SemanticDifferenceMetric, name="semantic_difference")