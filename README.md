# Hybrid Recommendation Engine

A sophisticated recommendation engine that combines collaborative filtering and implicit feedback approaches to provide accurate and diverse recommendations. This implementation uses state-of-the-art libraries to create a hybrid system that leverages the strengths of different recommendation techniques.

## Features

- **Multiple Recommendation Approaches**:
  - Collaborative Filtering (using Surprise)
    - Matrix Factorization with SVD
    - Explicit feedback handling
    - Rating prediction
  - Implicit Feedback (using Implicit)
    - Alternating Least Squares (ALS)
    - Implicit preference modeling
    - Fast computation for large datasets
  
- **Hybrid Approach Benefits**:
  - Better accuracy through ensemble methods
  - Handles both explicit and implicit feedback
  - Configurable weights for each approach
  - Scalable implementation

- **Professional Project Structure**:
  - Modular and maintainable code
  - Comprehensive test suite
  - Type hints and documentation
  - Configuration management
  - Example implementations
  - Interactive Jupyter notebook tutorial

## Project Structure

```
hybrid-recommender/
├── src/
│   └── hybrid_recommender/
│       ├── __init__.py
│       ├── recommender.py
│       └── config.py
├── tests/
│   └── test_recommender.py
├── examples/
│   └── basic_usage.py
├── notebooks/
│   └── recommender_tutorial.ipynb
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package:
```bash
pip install -e .
```

3. For Jupyter notebook tutorial:
```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/recommender_tutorial.ipynb
```

## Quick Start

```python
from hybrid_recommender import AdvancedHybridRecommender, RecommenderConfig

# Create a recommender with custom configuration
config = RecommenderConfig(
    method='all',
    weights={
        'surprise': 0.6,
        'implicit': 0.4
    }
)

# Initialize and train the recommender
recommender = AdvancedHybridRecommender(config)
recommender.fit(
    user_item_matrix=user_item_matrix,
    user_ids=user_ids,
    item_ids=item_ids
)

# Get recommendations
recommendations = recommender.recommend_items('user1', n_recommendations=5)
```

## Interactive Tutorial

We provide a comprehensive Jupyter notebook tutorial that demonstrates:
- Detailed usage examples
- Data preparation and visualization
- Different configuration options
- Performance analysis and evaluation
- Recommendation diversity analysis

To run the tutorial:
1. Install notebook dependencies: `pip install -e ".[notebook]"`
2. Start Jupyter: `jupyter notebook notebooks/recommender_tutorial.ipynb`

## Configuration

The `RecommenderConfig` class allows you to customize:
- Which recommendation methods to use
- Weights for each method in the ensemble
- Parameters for each underlying model
- Rating scales and other system parameters

Example configuration:
```python
config = RecommenderConfig(
    method='all',
    weights={'surprise': 0.6, 'implicit': 0.4},
    surprise_params={'n_factors': 100, 'n_epochs': 20},
    implicit_params={'factors': 50, 'regularization': 0.01}
)
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## License

MIT License - feel free to use this code for your own projects.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hybrid_recommender,
  title = {Hybrid Recommendation Engine},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/hybrid-recommender}
}
``` 