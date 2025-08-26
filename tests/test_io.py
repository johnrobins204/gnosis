import os
import tempfile
import pandas as pd
import pytest
from src.io import load_csv, write_dataframe, write_provenance

def test_write_and_load_csv(tmp_path):
    # Create a DataFrame and write it
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv_path = tmp_path / 'test.csv'
    write_dataframe(df, str(csv_path))
    # Load it back
    loaded = load_csv(str(csv_path))
    pd.testing.assert_frame_equal(df, loaded)

def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv('nonexistent_file.csv')

def test_write_provenance_creates_meta_file(tmp_path):
    # Write a dummy artifact
    artifact_path = tmp_path / 'artifact.txt'
    artifact_path.write_text('dummy')
    config = {'foo': 'bar'}
    meta_path = write_provenance(str(artifact_path), config)
    assert os.path.exists(meta_path)
    # Check meta file content
    import json
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assert meta['artifact'] == str(artifact_path)
    assert meta['config'] == config
