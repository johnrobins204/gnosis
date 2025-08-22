from abc import ABC, abstractmethod

class AnalyticsAddinBase(ABC):
    """
    Base class for all analytics add-ins.
    All add-ins must inherit from this and implement required methods.
    """
    @abstractmethod
    def run(self, data, **kwargs):
        """
        Run the add-in's main logic on the provided data.
        Args:
            data: Input data (e.g., DataFrame)
            **kwargs: Additional parameters
        Returns:
            Result of the add-in computation
        """
        pass

    @classmethod
    def from_manifest(cls, manifest):
        """
        Optional: Construct an add-in from a manifest dict.
        Args:
            manifest: Parsed manifest dict
        Returns:
            Instance of the add-in
        """
        return cls()
