# Analytics Add-in Authoring Guide

This guide explains how to create, register, and validate new analytics add-ins for the Gnosis analytics platform.

## 1. Create Your Add-in Class
- Inherit from `AnalyticsAddinBase`:

```python
from src.analytics.analytics_addin_base import AnalyticsAddinBase

class MyCustomAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        # Your logic here
        return ...
```

## 2. Write a Manifest File
- Create a YAML file (e.g., `my_addin.yaml`) with the following fields:

```yaml
name: "MyCustomAddin"
version: "0.1.0"
author: "Your Name <your.email@example.com>"
description: "Describe what your add-in does."
input_data_types:
  - DataFrame
  - columns:
      - name: "score"
        type: "float"
metrics:
  - name: "my_metric"
    description: "Describe your metric."
dependencies:
  - pandas>=1.3.0
entry_point: "my_module.MyCustomAddin"
```

## 3. Validate Your Add-in
- Use the provided validator to check your manifest:

```python
from src.analytics.analytics_addin_manifest_validator import validate_manifest
manifest = validate_manifest('my_addin.yaml')
```

- Use the input data validator to check your data:

```python
from src.analytics.input_data_validator import validate_input_data
validate_input_data(data, manifest)
```

## 4. Register and Discover Add-ins
- Place your manifest in the add-in manifests directory.
- Use the registry to discover and load add-ins:

```python
from src.analytics.addin_registry import AddinRegistry
reg = AddinRegistry()
reg.discover_from_manifests('path/to/manifests')
addin_cls = reg.get('MyCustomAddin')
addin = addin_cls()
```

## 5. Exception Handling
- Use the runner to execute add-ins safely:

```python
from src.analytics.addin_runner import run_addin_with_logging
result, error = run_addin_with_logging(addin, data)
```

## 6. Example Add-in and Manifest
- See `backlog/analytics_addin_manifest_example.yaml` for a template manifest.
- Example add-in class:

```python
class ExampleMetricAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        return data['score'].mean()
```

---
For more details, see the API docstrings in each module.
