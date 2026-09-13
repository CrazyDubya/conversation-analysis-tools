#!/usr/bin/env python3
"""Setup script for the consolidated conversation-analysis toolkit."""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith('#')]

setup(
    name='conversation-analysis',
    version='2.0.0',
    description='Consolidated toolkit for ChatGPT/Claude conversation archive analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Stephen Thompson',
    url='https://github.com/CrazyDubya/conversation-analysis-tools',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
    extras_require={
        'optional': [
            'anthropic>=0.18.0',
            'openai>=1.0.0',
            'sqlparse>=0.4.0',
            'python-louvain>=0.16',
            'pyvis>=0.3.0',
            'graphviz>=0.20.0',
            'Pillow>=9.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
        ],
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'analyze-content=conversation_analysis.pipeline.pipeline:main',
        ],
    },
)
