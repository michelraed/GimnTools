#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script para a biblioteca GimnTools
"""

from setuptools import setup, find_packages
import os

# Ler o README para a descrição longa
def read_file(filename):
    """Lê arquivo e retorna seu conteúdo"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Informações da versão
VERSION = "1.0.0"

# Dependências principais
INSTALL_REQUIRES = [
    "numpy>=1.20.0",
    "matplotlib>=3.3.0",
    "scipy>=1.7.0",
    "numba>=0.54.0",
    "SimpleITK>=2.1.0",
    "itk>=5.2.0",
    "Pillow>=8.0.0",
    "scikit-image>=0.18.0",
    "h5py>=3.1.0",
    "tqdm>=4.60.0",
    "pandas>=1.3.0",
    "awkward>=1.2.0",
    "uproot>=5.0.0",
]

# Dependências opcionais para desenvolvimento
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
        'mypy>=0.910',
    ],
    'jupyter': [
        'jupyter>=1.0.0',
        'ipywidgets>=7.6.0',
        'plotly>=5.0.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'sphinx-autodoc-typehints>=1.12.0',
        'myst-parser>=0.15.0',
    ]
}

# Todas as dependências extras
EXTRAS_REQUIRE['all'] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

setup(
    name="GimnTools",
    version=VERSION,
    author="Desenvolvedor GimnTools",
    author_email="michelraed@gmail.com",
    description="Biblioteca Python para processamento de imagens médicas e reconstrução tomográfica",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/GimnTools",
    project_urls={
        "Bug Reports": "https://github.com/usuario/GimnTools/issues",
        "Source": "https://github.com/usuario/GimnTools",
        "Documentation": "https://gimntools.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    package_data={
        "GimnTools": [
            "data/*.json",
            "data/*.yaml",
            "examples/*.py",
            "examples/data/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "gimntools-info=GimnTools.bin.info:main",
            "gimntools-convert=GimnTools.bin.convert:main",
            "gimntools-reconstruct=GimnTools.bin.reconstruct:main",
        ]
    },
    zip_safe=False,
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "coverage>=5.5",
        "hypothesis>=6.0.0",
    ],
    license="MIT",
    platforms=["any"],
    keywords=[
        "medical imaging",
        "tomography",
        "reconstruction",
        "sinogram", 
        "DICOM",
        "image processing",
        "PET",
        "SPECT"
    ],
)
