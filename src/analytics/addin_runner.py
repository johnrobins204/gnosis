import logging
from .analytics_addin_base import AnalyticsAddinBase

def run_addin_with_logging(addin: AnalyticsAddinBase, data, **kwargs):
    """
    Run an add-in with exception handling and logging.
    Returns (result, error):
      - result: output of add-in if successful, else None
      - error: exception instance if failed, else None
    """
    logger = logging.getLogger("analytics.addin")
    try:
        result = addin.run(data, **kwargs)
        return result, None
    except Exception as e:
        logger.error(f"Add-in {addin.__class__.__name__} failed: {e}", exc_info=True)
        return None, e
