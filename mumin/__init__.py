__version__ = '0.0.0'

# Set up logging
import logging
fmt = '%(asctime)s [%(levelname)s - %(name)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('newspaper').setLevel(logging.CRITICAL)

from .dataset import MuminDataset  # noqa
