# gnosis Prompt Workbench (gnosisPWB)

A comprehensive toolkit for prompt engineering research, experimentation, and optimization.

## Overview

gnosis Prompt Workbench (gnosisPWB) is an open-source toolkit designed for AI researchers and developers working with large language models. It provides a structured environment for systematic prompt engineering and evaluation.

### Core Capabilities

- **Prompt Research**: Systematically study how different prompting techniques affect model outputs
- **Performance Analysis**: Measure and compare response quality across models and prompt variations
- **Optimization Workflows**: Refine prompts through iterative testing and validation
- **Experiment Tracking**: Document your prompt engineering journey with reproducible results

## Installation

```bash
# Clone the repository
git clone https://github.com/johnrobins204/gnosis.git
cd gnosis

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Who Should Use gnosisPWB

gnosisPWB is particularly valuable for:

- **AI Researchers** conducting systematic studies of prompt engineering techniques
- **Data Science Teams** validating and optimizing prompts for production use
- **LLM Engineers** seeking to document and justify prompt design decisions
- **ML Operations Teams** managing prompt quality across multiple models and versions

The tool shines in scenarios requiring methodical prompt improvement with clear evaluation criteria and documented decision-making.

## Key Features

- **Model Abstraction Layer**: Interact with different LLMs through a unified interface
- **Prompt Templates**: Create, store, and version your prompts
- **Automated Evaluation**: Assess responses against predefined criteria
- **Visualization Tools**: Compare results with intuitive charts and reports
- **Experiment Tracking**: Maintain a history of your prompt engineering iterations

## Usage Examples

### Basic Inference

```python
from src.inference import run_inference
from src.types import ModelResponse

# Run a simple inference
response = run_inference(
    prompt="Explain the concept of prompt engineering in simple terms",
    model="gpt-3.5-turbo"
)
print(response.content)
```

### Comparing Prompts

```python
from src.analyst import compare_prompts
from src.types import PromptVariation

# Define prompt variations
variations = [
    PromptVariation(
        name="basic",
        content="Write a short poem about AI"
    ),
    PromptVariation(
        name="detailed",
        content="Write a short poem about artificial intelligence. "
                "Use imagery related to knowledge and learning."
    )
]

# Compare results across variations
results = compare_prompts(variations, model="gpt-3.5-turbo")
```

### Running Evaluations

```python
from src.judge import evaluate_response
from src.inference import run_inference

# Generate a response
prompt = "Explain how a transformer neural network works"
response = run_inference(prompt, model="gpt-3.5-turbo")

# Evaluate the response
scores = evaluate_response(
    response,
    criteria=["accuracy", "clarity", "completeness"]
)
```

## Where gnosisPWB Fits

gnosisPWB bridges the gap between ad-hoc prompt experimentation and production-ready prompt engineering. While many tools focus on either building LLM applications (LangChain), versioning prompts (PromptLayer), or general ML experiment tracking (W&B), gnosisPWB provides an integrated environment specifically for systematic prompt research and optimization.

### Comparison with Existing Tools

| Feature | gnosisPWB | LangChain | PromptLayer | Promptfoo | W&B |
|---------|:---------:|:---------:|:-----------:|:---------:|:---:|
| Prompt Versioning | ✅ | ❌ | ✅ | ✅ | ✅ |
| Response Evaluation | ✅ | ❌ | ❌ | ✅ | ❌ |
| Analytics Dashboard | ✅ | ❌ | ✅ | ❌ | ✅ |
| Multi-model Support | ✅ | ✅ | ✅ | ✅ | ✅ |
| Integrated Research Workflow | ✅ | ❌ | ❌ | ❌ | ❌ |
| Application Building | ❌ | ✅ | ❌ | ❌ | ❌ |

*Note: This comparison reflects our current understanding and may evolve as tools in this space rapidly develop.*

## Project Status and Roadmap

gnosisPWB is currently in beta. Core functionality is implemented and usable, but the API may evolve based on user feedback.

### Roadmap

#### Immediate Priority
- Complete analytics module for comprehensive result analysis

#### Short-term
- Database integration for result capture using lightweight local open data server
- Test question bank integration with CRUD APIs
- Experiment "fingerprinting" for reliable reproduction

#### Mid-term
- Export/import study configurations via standardized data structure
- Local model performance visibility through oLlama integration
- Improved visualization components and exportable reports

#### Long-term
- Evaluate potential value of GUI integration
- Integration with popular model providers via unified API

#### Future Explorations
- "Agentic" study model for chaining prompts (chain-of-thought, etc.)
- Advanced metrics and benchmarking against industry standards

We welcome feature requests and prioritize development based on community needs.
## Limitations and Considerations

- gnosisPWB is designed for prompt research and optimization, not as a complete application development framework
- For production deployments, you may want to export optimized prompts to a dedicated serving infrastructure
- The tool currently focuses on text-based LLM interactions, with limited support for multimodal models
- Response evaluation relies on user-defined criteria and may require domain expertise to configure effectively

## Project Structure

```
gnosis/
├── .github/workflows/  # CI configuration
├── scripts/            # Utility scripts and tools
├── src/                # Main package
│   ├── models/         # Model implementations and adapters
│   ├── tests/          # Test suite
│   ├── analyst.py      # Analytics and comparison tools
│   ├── analytics.py    # Data analysis utilities
│   ├── cli.py          # Command-line interface
│   ├── inference.py    # Inference pipeline
│   ├── io.py           # Input/output utilities
│   ├── judge.py        # Evaluation components
│   ├── orchestrator.py # Workflow coordination
│   ├── types.py        # Data structures
│   └── ...
├── .gitignore          # Git ignore file
├── pyproject.toml      # Project metadata
├── README.md           # This file
├── requirements.txt    # Dependencies
└── setup.cfg           # Package configuration
```

## Development

To run tests:

```bash
# Run all tests
pytest

# Run specific tests
pytest src/tests/test_types.py

# Run with coverage report
pytest --cov=src
```

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was inspired by the growing need for systematic prompt engineering tools
- Thanks to all contributors who have helped shape gnosis Prompt Workbench

---

_**gnosis Prompt Workbench** — γνῶσις διά ἀκρίβειας (gnosis dia akriveias) — Knowledge through rigour_