import os
import pytest
from src.analytics.analytics_addin_manifest_validator import validate_manifest, ManifestValidationError

TEST_MANIFEST_DIR = os.path.join(os.path.dirname(__file__), '../../backlog')

VALID_MANIFEST = os.path.join(TEST_MANIFEST_DIR, 'analytics_addin_manifest_example.yaml')

INVALID_MANIFEST_MISSING = os.path.join(TEST_MANIFEST_DIR, 'invalid_manifest_missing.yaml')
INVALID_MANIFEST_TYPE = os.path.join(TEST_MANIFEST_DIR, 'invalid_manifest_type.yaml')

def test_valid_manifest():
    manifest = validate_manifest(VALID_MANIFEST)
    assert manifest['name'] == 'ExampleMetricAddin'
    assert 'metrics' in manifest

def test_missing_required_fields(tmp_path):
    # Remove 'name' from manifest
    with open(VALID_MANIFEST) as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.startswith('name:')]
    test_path = tmp_path / 'missing_name.yaml'
    with open(test_path, 'w') as f:
        f.writelines(lines)
    with pytest.raises(ManifestValidationError) as e:
        validate_manifest(str(test_path))
    assert 'Missing required fields' in str(e.value)

def test_invalid_field_type(tmp_path):
    # Change 'version' to a list
    with open(VALID_MANIFEST) as f:
        lines = f.readlines()
    lines = [l if not l.startswith('version:') else 'version: [0.1.0]\n' for l in lines]
    test_path = tmp_path / 'invalid_version_type.yaml'
    with open(test_path, 'w') as f:
        f.writelines(lines)
    with pytest.raises(ManifestValidationError) as e:
        validate_manifest(str(test_path))
    assert "'version' must be a string" in str(e.value)

def test_invalid_metrics_structure(tmp_path):
    # Set metrics to a list of strings only (syntactically valid YAML, semantically invalid)
    with open(VALID_MANIFEST) as f:
        lines = f.readlines()
    idx = [i for i, l in enumerate(lines) if l.startswith('metrics:')][0]
    new_lines = lines[:idx+1]
    new_lines.append('  - not_a_dict\n')
    # Skip all original metric entries (which start with '  -' or are blank)
    i = idx + 1
    while i < len(lines) and (lines[i].startswith('  -') or lines[i].strip() == '' or lines[i].startswith('    ')):
        i += 1
    new_lines.extend(lines[i:])
    test_path = tmp_path / 'invalid_metrics.yaml'
    with open(test_path, 'w') as f:
        f.writelines(new_lines)
    # Print the generated YAML for debugging
    print(f"Generated invalid metrics YAML:\n{open(test_path).read()}")
    with pytest.raises(ManifestValidationError) as e:
        validate_manifest(str(test_path))
    assert "Each metric must be a dict with a 'name' field" in str(e.value)
