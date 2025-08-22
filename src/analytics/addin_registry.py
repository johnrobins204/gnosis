import os
import importlib
import yaml
from .analytics_addin_manifest_validator import validate_manifest, ManifestValidationError
from .analytics_addin_base import AnalyticsAddinBase

class AddinRegistry:
    """
    Registry and discovery for analytics add-ins.
    """
    def __init__(self):
        self._registry = {}

    def register(self, name, addin_cls):
        if not issubclass(addin_cls, AnalyticsAddinBase):
            raise TypeError(f"{addin_cls} must inherit from AnalyticsAddinBase")
        self._registry[name] = addin_cls

    def get(self, name):
        return self._registry.get(name)

    def discover_from_manifests(self, manifest_dir):
        """
        Scan a directory for add-in manifests, validate, and register add-ins.
        """
        for fname in os.listdir(manifest_dir):
            if not fname.endswith('.yaml'):
                continue
            manifest_path = os.path.join(manifest_dir, fname)
            try:
                manifest = validate_manifest(manifest_path)
                entry_point = manifest['entry_point']
                module_name, class_name = entry_point.rsplit('.', 1)
                module = importlib.import_module(module_name)
                addin_cls = getattr(module, class_name)
                self.register(manifest['name'], addin_cls)
            except (ManifestValidationError, ImportError, AttributeError, TypeError) as e:
                # Could log or collect errors here
                continue

    def list_addins(self):
        return list(self._registry.keys())
