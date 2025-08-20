import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.analytics.visualization import run_from_config, Visualization

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "model": ["Model A", "Model B", "Model A", "Model B"],
        "prompt": ["Prompt 1", "Prompt 1", "Prompt 2", "Prompt 2"],
        "score": [0.8, 0.7, 0.9, 0.6],
        "accuracy": [0.9, 0.8, 0.7, 0.6]
    })

@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_run_from_config_missing_keys():
    """Test handling of missing config keys."""
    # Test missing input_csv
    result = run_from_config({"output_dir": "output", "visualization_type": "comparative"})
    assert result["success"] is False
    assert "missing config key: input_csv" in result["error"]
    
    # Test missing output_dir
    result = run_from_config({"input_csv": "input.csv", "visualization_type": "comparative"})
    assert result["success"] is False
    assert "missing config key: output_dir" in result["error"]
    
    # Test missing visualization_type
    result = run_from_config({"input_csv": "input.csv", "output_dir": "output"})
    assert result["success"] is False
    assert "missing config key: visualization_type" in result["error"]

@patch("src.analytics.visualization.load_csv")
def test_run_from_config_load_error(mock_load):
    """Test handling of errors when loading input file."""
    # Setup mock to raise exception
    mock_load.side_effect = Exception("File not found")
    
    result = run_from_config({
        "input_csv": "nonexistent.csv",
        "output_dir": "output",
        "visualization_type": "comparative"
    })
    
    assert result["success"] is False
    assert "failed to load input_csv" in result["error"]

@patch("src.analytics.visualization.load_csv")
def test_run_from_config_empty_data(mock_load):
    """Test handling of empty input data."""
    # Setup mock to return empty DataFrame
    mock_load.return_value = pd.DataFrame()
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": "output",
        "visualization_type": "comparative"
    })
    
    assert result["success"] is False
    assert "input data is empty" in result["error"]

@patch("src.analytics.visualization.load_csv")
@patch("src.analytics.visualization.Visualization")
def test_run_from_config_comparative(mock_visualization_class, mock_load, sample_data, temp_directory):
    """Test comparative visualization."""
    # Setup mocks
    mock_load.return_value = sample_data
    mock_viz = MagicMock()
    mock_visualization_class.return_value = mock_viz
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": temp_directory,
        "visualization_type": "comparative",
        "metric_name": "score"
    })
    
    # Verify success
    assert result["success"] is True
    assert len(result["artifacts"]) == 1
    assert "comparative_score.png" in result["artifacts"][0]
    
    # Verify visualization was created
    mock_viz.plot_comparative_metrics.assert_called_once_with(
        data=ANY,
        metric_name="score",
        output_path=ANY
    )

@patch("src.analytics.visualization.load_csv")
@patch("src.analytics.visualization.Visualization")
def test_run_from_config_statistical(mock_visualization_class, mock_load, sample_data, temp_directory):
    """Test statistical visualization."""
    # Setup mocks
    mock_load.return_value = sample_data
    mock_viz = MagicMock()
    mock_visualization_class.return_value = mock_viz
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": temp_directory,
        "visualization_type": "statistical",
        "metric_name": "score"
    })
    
    # Verify success
    assert result["success"] is True
    assert len(result["artifacts"]) == 1
    assert "statistical_score.png" in result["artifacts"][0]
    
    # Verify visualization was created
    mock_viz.plot_statistical_visualizations.assert_called_once_with(
        data=ANY,
        metric_name="score",
        output_path=ANY
    )

@patch("src.analytics.visualization.load_csv")
@patch("src.analytics.visualization.Visualization")
def test_run_from_config_radar(mock_visualization_class, mock_load, sample_data, temp_directory):
    """Test radar chart visualization."""
    # Setup mocks
    mock_load.return_value = sample_data
    mock_viz = MagicMock()
    mock_visualization_class.return_value = mock_viz
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": temp_directory,
        "visualization_type": "radar"
    })
    
    # Verify success
    assert result["success"] is True
    assert len(result["artifacts"]) == 1
    assert "radar_chart.png" in result["artifacts"][0]
    
    # Verify visualization was created
    mock_viz.plot_radar_chart.assert_called_once()

@patch("src.analytics.visualization.load_csv")
def test_run_from_config_unknown_type(mock_load, sample_data):
    """Test handling of unknown visualization type."""
    # Setup mock
    mock_load.return_value = sample_data
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": "output",
        "visualization_type": "unknown"
    })
    
    assert result["success"] is False
    assert "unknown visualization_type" in result["error"]

@patch("src.analytics.visualization.load_csv")
def test_run_from_config_missing_metric(mock_load, sample_data):
    """Test handling of missing metric name for visualizations that require it."""
    # Setup mock
    mock_load.return_value = sample_data
    
    # Test comparative without metric_name
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": "output",
        "visualization_type": "comparative"
    })
    
    assert result["success"] is False
    assert "metric_name is required" in result["error"]
    
    # Test statistical without metric_name
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": "output",
        "visualization_type": "statistical"
    })
    
    assert result["success"] is False
    assert "metric_name is required" in result["error"]

@patch("src.analytics.visualization.load_csv")
def test_run_from_config_radar_no_numeric(mock_load):
    """Test handling of radar chart with no numeric columns."""
    # Setup mock to return DataFrame with no numeric columns
    mock_load.return_value = pd.DataFrame({
        "model": ["A", "B"],
        "prompt": ["X", "Y"]
    })
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_dir": "output",
        "visualization_type": "radar"
    })
    
    assert result["success"] is False
    assert "no numeric columns found" in result["error"]

# Tests for the Visualization class
def test_visualization_init():
    """Test initialization with different themes."""
    # Test default theme
    viz = Visualization()
    assert viz.theme == "academic"
    
    # Test custom theme
    viz = Visualization(theme="custom")
    assert viz.theme == "custom"

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_comparative_metrics(mock_close, mock_savefig):
    """Test comparative metrics plot."""
    # Create test data
    data = [
        {"model": "Model A", "prompt": "Prompt 1", "score": 0.8},
        {"model": "Model B", "prompt": "Prompt 1", "score": 0.7}
    ]
    
    # Create visualization and call method
    viz = Visualization()
    viz.plot_comparative_metrics(
        data=data,
        metric_name="score",
        output_path="test.png"
    )
    
    # Verify savefig was called
    mock_savefig.assert_called_once_with("test.png")
    
    # Verify close was called to prevent displaying
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_statistical_visualizations(mock_close, mock_savefig):
    """Test statistical visualizations."""
    # Create test data
    data = [
        {"model": "Model A", "prompt": "Prompt 1", "score": 0.8},
        {"model": "Model B", "prompt": "Prompt 1", "score": 0.7},
        {"model": "Model A", "prompt": "Prompt 2", "score": 0.9},
        {"model": "Model B", "prompt": "Prompt 2", "score": 0.6}
    ]
    
    # Create visualization and call method
    viz = Visualization()
    viz.plot_statistical_visualizations(
        data=data,
        metric_name="score",
        output_path="test.png"
    )
    
    # Verify savefig was called
    mock_savefig.assert_called_once_with("test.png")
    
    # Verify close was called
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_radar_chart(mock_close, mock_savefig):
    """Test radar chart creation."""
    # Create test data
    metrics = [0.8, 0.6, 0.9, 0.7]
    labels = ["Accuracy", "Relevance", "Coherence", "Helpfulness"]
    
    # Create visualization and call method
    viz = Visualization()
    viz.plot_radar_chart(
        metrics=metrics,
        labels=labels,
        output_path="test.png"
    )
    
    # Verify savefig was called
    mock_savefig.assert_called_once_with("test.png")
    
    # Verify close was called
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
def test_export_figure(mock_savefig):
    """Test exporting a figure."""
    # Create visualization and call method
    viz = Visualization()
    viz.export_figure("test.png", format="png")
    
    # Verify savefig was called with correct params
    mock_savefig.assert_called_once_with("test.png", format="png")

def test_plot_without_output_path():
    """Test plotting without output path."""
    # Create test data
    data = [
        {"model": "Model A", "prompt": "Prompt 1", "score": 0.8},
        {"model": "Model B", "prompt": "Prompt 1", "score": 0.7}
    ]
    
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        # Create visualization and call method without output path
        viz = Visualization()
        viz.plot_comparative_metrics(
            data=data,
            metric_name="score"
        )
        
        # Verify savefig was not called
        mock_savefig.assert_not_called()