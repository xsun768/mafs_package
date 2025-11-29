from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mafs",
    version="0.1.0",
    author="Xiaoyan",
    author_email="xsun768@aucklanduni.ac.nz",
    description="Multi-head Attention-based Feature Selection for high-dimensional biomedical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xsun768/mafs_package",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # 核心依赖
    install_requires=[
        "numpy>=1.20.0,<3.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "scikit-learn>=1.0.0",
    ],
    
    # 可选依赖
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "ipython>=8.0",
            "optuna>=3.0.0",
            "tqdm>=4.60.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "plotnine>=0.10.0",
        ],
    },
    
    # 包含数据文件（如果你有）
    include_package_data=True,
    
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
