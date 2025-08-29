
from logging_config import get_logger
from analytics.analytics_addin_base import AnalyticsAddinBase

def run_addin_with_logging(addin: AnalyticsAddinBase, data, **kwargs):
  """
  Run an add-in with exception handling and logging.
  Returns (result, error):
    - result: output of add-in if successful, else None
    - error: exception instance if failed, else None
  """
  logger = get_logger("analytics.addin")
  try:
    result = addin.run(data, **kwargs)
    return result, None
  except (ValueError, RuntimeError) as e:
    logger.error(f"Add-in {addin.__class__.__name__} failed with a known error: {e}", exc_info=True)
    return None, e
  except Exception as e:
    # Catch-all to prevent addin failures from crashing the pipeline
    logger.error(f"Add-in {addin.__class__.__name__} failed with an unexpected error: {e}", exc_info=True)
    return None, e
