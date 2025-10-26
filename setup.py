#!/usr/bin/env python3
"""Setup script for Content Analysis Pipeline.

Installs all required dependencies and sets up the package.
"""

from setuptools import setup, find_packages
import os

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
requirements = [
    'numpy>=1.21.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'PyYAML>=5.4.0',
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
]

setup(
    name='conversation-analysis-tools',
    version='1.0.0',
    description='Content analysis pipeline for scraped research content',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Stephen Thompson',
    author_email='',
    url='https://github.com/CrazyDubya/conversation-analysis-tools',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'analyze-content=pipeline.pipeline:main',
        ],
    },
)
