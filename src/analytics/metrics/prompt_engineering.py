from src.analytics.metrics.base import Metric, MetricRegistry
from transformers import GPT2Tokenizer

class TokenUtilizationMetric(Metric):
    """Calculate token utilization metrics (input/output ratio, efficiency)."""

    def __init__(self, tokenizer_model="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model)

    def calculate(self, data):
        """Calculate token utilization metrics."""
        if not data["prompts"] or not data["responses"]:
            raise ValueError("Prompts and responses cannot be empty.")
        prompts, responses = data["prompts"], data["responses"]
        prompt_tokens = [len(self.tokenizer.tokenize(prompt)) for prompt in prompts]
        response_tokens = [len(self.tokenizer.tokenize(response)) for response in responses]
        utilization = [resp / prompt if prompt > 0 else 0 for prompt, resp in zip(prompt_tokens, response_tokens)]
        return {"prompt_tokens": prompt_tokens, "response_tokens": response_tokens, "utilization": utilization}

    def name(self):
        return "TokenUtilization"

class InstructionAdherenceMetric(Metric):
    """Score instruction adherence based on predefined criteria."""

    def calculate(self, data):
        """Calculate instruction adherence scores."""
        instructions, responses = data["instructions"], data["responses"]
        adherence_scores = [
            1 if instruction.lower() in response.lower() else 0
            for instruction, response in zip(instructions, responses)
        ]
        return {"adherence_scores": adherence_scores, "average_adherence": sum(adherence_scores) / len(adherence_scores)}

    def name(self):
        return "InstructionAdherence"

class PerformanceCostRatioMetric(Metric):
    """Calculate performance/cost ratio metrics."""

    def calculate(self, data):
        """Calculate performance/cost ratios."""
        performance_scores, costs = data["performance_scores"], data["costs"]
        ratios = [perf / cost if cost > 0 else 0 for perf, cost in zip(performance_scores, costs)]
        return {"performance_cost_ratios": ratios}

    def name(self):
        return "PerformanceCostRatio"

# Register metrics
MetricRegistry.register(TokenUtilizationMetric)
MetricRegistry.register(InstructionAdherenceMetric)
MetricRegistry.register(PerformanceCostRatioMetric)