"""
Setup script for Micro_GUI - Microstructure Analysis GUI Application
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='micro-gui',
    version='0.1.0',
    author='Hamed Amiri',
    author_email='amiiri.hamed@gmail.com',
    description='GUI Application for Statistical Microstructure Characterisation and Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hamediut/SMiCA',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'micro-gui=micro_gui.main:main',
        ],
    },
    include_package_data=True,
    keywords='microstructure analysis correlation-functions materials-science image-processing',
    project_urls={
        'Bug Reports': 'https://github.com/hamediut/SMiCA/issues',
        'Source': 'https://github.com/hamediut/SMiCA',
    },
)
