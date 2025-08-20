import pytest
import json
import os
import tempfile
import hashlib
from src.analytics.experiment_tracker import ExperimentTracker

@pytest.fixture
def temp_dir():
    """Create a temporary directory for experiment data."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_experiment_tracker_init():
    """Test ExperimentTracker initialization."""
    tracker = ExperimentTracker()
    # Check that storage directory exists
    assert os.path.exists(tracker.storage_path)
    # Default path should be 'experiments/'
    assert tracker.storage_path == "experiments/"

def test_generate_fingerprint():
    """Test generating a unique fingerprint for an experiment."""
    tracker = ExperimentTracker()
    
    config1 = {"model": "gpt-4", "temperature": 0.7}
    config2 = {"model": "gpt-4", "temperature": 0.8}
    
    # Same config should produce same fingerprint
    fp1a = tracker.generate_fingerprint(config1)
    fp1b = tracker.generate_fingerprint(config1)
    assert fp1a == fp1b
    
    # Different configs should produce different fingerprints
    fp2 = tracker.generate_fingerprint(config2)
    assert fp1a != fp2
    
    # Order shouldn't matter for fingerprint
    config3 = {"temperature": 0.7, "model": "gpt-4"}
    fp3 = tracker.generate_fingerprint(config3)
    assert fp1a == fp3

def test_save_and_load_results(temp_dir):
    """Test saving and loading experiment results."""
    tracker = ExperimentTracker(storage_path=temp_dir)
    
    # Generate a fingerprint
    config = {"model": "gpt-4", "dataset": "eval-set-1"}
    fingerprint = tracker.generate_fingerprint(config)
    
    # Create some results
    results = {
        "accuracy": 0.85,
        "f1_score": 0.82,
        "examples": ["good example", "bad example"]
    }
    
    # Save the results
    tags = ["production", "high-quality"]
    tracker.save_results(fingerprint, results, tags)
    
    # Check that the file exists
    file_path = os.path.join(temp_dir, f"{fingerprint}.json")
    assert os.path.exists(file_path)
    
    # Load the results
    loaded_data = tracker.load_results(fingerprint)
    
    # Verify the data
    assert loaded_data["results"] == results
    assert loaded_data["tags"] == tags

def test_list_experiments(temp_dir):
    """Test listing all experiments."""
    tracker = ExperimentTracker(storage_path=temp_dir)
    
    # Create a few experiments
    configs = [
        {"model": "model1", "dataset": "dataset1"},
        {"model": "model2", "dataset": "dataset1"},
        {"model": "model1", "dataset": "dataset2"}
    ]
    
    fingerprints = []
    for config in configs:
        fp = tracker.generate_fingerprint(config)
        fingerprints.append(fp)
        tracker.save_results(fp, {"score": 0.5})
    
    # List all experiments
    experiments = tracker.list_experiments()
    
    # Verify all fingerprints are listed
    assert len(experiments) == len(fingerprints)
    for fp in fingerprints:
        assert fp in experiments

def test_filter_experiments(temp_dir):
    """Test filtering experiments by tag."""
    tracker = ExperimentTracker(storage_path=temp_dir)
    
    # Create experiments with different tags
    configs = [
        {"model": "model1", "params": {"a": 1}},
        {"model": "model2", "params": {"a": 2}},
        {"model": "model3", "params": {"a": 3}}
    ]
    
    # Save with different tags
    fp1 = tracker.generate_fingerprint(configs[0])
    tracker.save_results(fp1, {"score": 0.7}, tags=["production", "good"])
    
    fp2 = tracker.generate_fingerprint(configs[1])
    tracker.save_results(fp2, {"score": 0.8}, tags=["development", "good"])
    
    fp3 = tracker.generate_fingerprint(configs[2])
    tracker.save_results(fp3, {"score": 0.6}, tags=["production", "needs-improvement"])
    
    # Filter by "production" tag
    production_exps = tracker.filter_experiments("production")
    assert len(production_exps) == 2
    assert fp1 in production_exps
    assert fp3 in production_exps
    
    # Filter by "good" tag
    good_exps = tracker.filter_experiments("good")
    assert len(good_exps) == 2
    assert fp1 in good_exps
    assert fp2 in good_exps
    
    # Filter by tag that doesn't exist
    empty_exps = tracker.filter_experiments("nonexistent")
    assert len(empty_exps) == 0

def test_load_nonexistent_results(temp_dir):
    """Test loading results for a nonexistent experiment."""
    tracker = ExperimentTracker(storage_path=temp_dir)
    
    with pytest.raises(FileNotFoundError):
        tracker.load_results("nonexistent-fingerprint")