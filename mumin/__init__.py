__version__ = '0.0.0'

# Set up logging
import logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)

from .dataset import MuminDataset  # noqa
