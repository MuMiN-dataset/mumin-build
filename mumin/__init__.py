__version__ = '1.5.0'

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
logging.getLogger('numexpr').disabled = True
logging.getLogger('numexpr').propagate = False
logging.getLogger('bs4').disabled = True
logging.getLogger('bs4').propagate = False
logging.getLogger('transformers').disabled = True
logging.getLogger('transformers').propagate = False

from .dataset import MuminDataset  # noqa
from .dgl import load_dgl_graph, save_dgl_graph  # noqa
