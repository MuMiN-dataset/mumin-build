from setuptools import setup, find_packages
from pathlib import Path
from bump_version import get_current_version


# Set up extras
DGL_EXTRAS = ['dgl>=0.6.1', 'torch>=1.9.0']
EMBEDDINGS_EXTRAS = ['torch>=1.9.0', 'transformers>=4.10.0']
ALL_EXTRAS = DGL_EXTRAS + EMBEDDINGS_EXTRAS


setup(name='mumin',
      version=get_current_version(return_tuple=False),
      description='',
      long_description=Path('README.md').read_text(),
      long_description_content_type='text/markdown',
      url='https://github.com/CLARITI-REPHRAIN/mumin-build',
      author='Dan Saattrup Nielsen and Ryan McConville',
      author_email='dan.nielsen@bristol.ac.uk',
      license='MIT',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'],
      packages=find_packages(exclude=('tests',)),
      include_package_data=True,
      install_requires=['pandas>=1.3.0',
                        'newspaper3k>=0.2.8',
                        'requests>=2.26.0',
                        'tqdm>=4.62.2',
                        'opencv-python>=4.5.3.56',
                        'timeout-decorator>=0.5.0',
                        'wget>=3.2'],
      extras_require=dict(dgl=DGL_EXTRAS,
                          embeddings=EMBEDDINGS_EXTRAS,
                          all=ALL_EXTRAS))
