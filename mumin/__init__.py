__version__ = '0.0.0'

# Set up logging
import logging
fmt = '%(asctime)s [%(levelname)s - %(name)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logging.getLogger('urllib3').disabled = True
logging.getLogger('newspaper').disabled = True
logging.getLogger('jieba').disabled = True

from .dataset import MuminDataset  # noqa
