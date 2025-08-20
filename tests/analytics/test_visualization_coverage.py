import pytest
from unittest.mock import patch
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from src.analytics.visualization import Visualization

@pytest.fixture
def sample_data():
    """Sample data for visualization testing."""
    return [
        {"model": "Model A", "prompt": "Prompt 1", "score": 0.8, "accuracy": 0.9},
        {"model": "Model B", "prompt": "Prompt 1", "score": 0.7, "accuracy": 0.8},
        {"model": "Model A", "prompt": "Prompt 2", "score": 0.9, "accuracy": 0.7},
        {"model": "Model B", "prompt": "Prompt 2", "score": 0.6, "accuracy": 0.6}
    ]

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_comparative_metrics(mock_close, mock_savefig, sample_data):
    viz = Visualization()
    viz.plot_comparative_metrics(
        data=sample_data,
        metric_name="score",
        output_path="test_comparative.png"
    )
    mock_savefig.assert_called_once_with("test_comparative.png")
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_statistical_visualizations(mock_close, mock_savefig, sample_data):
    viz = Visualization()
    viz.plot_statistical_visualizations(
        data=sample_data,
        metric_name="accuracy",
        output_path="test_statistical.png"
    )
    mock_savefig.assert_called_once_with("test_statistical.png")
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_radar_chart(mock_close, mock_savefig):
    viz = Visualization()
    metrics = [0.8, 0.6, 0.9, 0.7]
    labels = ["Accuracy", "Relevance", "Coherence", "Helpfulness"]
    viz.plot_radar_chart(
        metrics=metrics,
        labels=labels,
        output_path="test_radar.png"
    )
    mock_savefig.assert_called_once_with("test_radar.png")
    mock_close.assert_called_once()