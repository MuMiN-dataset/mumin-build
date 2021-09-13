__version__ = '0.1.3'

# Set up logging
import logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logging.getLogger('urllib3').disabled = True
logging.getLogger('urllib3').propagate = False
logging.getLogger('newspaper').disabled = True
logging.getLogger('newspaper').propagate = False
logging.getLogger('jieba').disabled = True
logging.getLogger('jieba').propagate = False

from .dataset import MuminDataset  # noqa
