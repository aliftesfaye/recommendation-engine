"""
Hybrid Recommender System
========================

A sophisticated recommendation engine that combines multiple approaches:
- Collaborative Filtering (using Surprise)
- Content-Based Filtering (using LightFM)
- Implicit Feedback (using Implicit)
"""

from hybrid_recommender.recommender import AdvancedHybridRecommender
from hybrid_recommender.config import RecommenderConfig

__version__ = "0.1.0"
__all__ = ["AdvancedHybridRecommender", "RecommenderConfig"] 