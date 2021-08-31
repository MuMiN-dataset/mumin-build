from setuptools import setup, find_packages
from pathlib import Path
from bump_version import get_current_version

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
      install_requires=['pandas>=1.3.2'],
      extras_require=dict(dgl=['dgl>=0.6.1'],
                          pyg=['torch-scatter>=2.0.8',
                               'torch-sparse>=0.6.11',
                               'torch-spline-conv>=1.2.1',
                               'torch-geometric>=1.7.2']))
