from setuptools import setup, find_packages

setup(
    name="hybrid_recommender",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "implicit>=0.7.0",
        "tensorflow>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "ipywidgets>=7.0.0",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid recommendation engine combining multiple approaches",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="recommender-systems machine-learning hybrid-recommendations",
    url="https://github.com/yourusername/hybrid-recommender",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 