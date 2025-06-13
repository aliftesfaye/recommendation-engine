"""Tests for the hybrid recommender system."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from hybrid_recommender import AdvancedHybridRecommender, RecommenderConfig

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample interaction data
    interactions = [
        ('user1', 'item1', 5.0),
        ('user1', 'item2', 3.0),
        ('user1', 'item4', 1.0),
        ('user1', 'item6', 2.0),
        ('user2', 'item1', 4.0),
        ('user2', 'item4', 1.0),
        ('user2', 'item5', 2.0),
        ('user3', 'item1', 1.0),
        ('user3', 'item2', 2.0),
        ('user3', 'item3', 3.0),
        ('user3', 'item4', 4.0),
        ('user4', 'item3', 4.0),
        ('user4', 'item4', 3.0),
        ('user4', 'item5', 5.0),
        ('user4', 'item6', 1.0),
        ('user5', 'item1', 2.0),
        ('user5', 'item2', 1.0),
        ('user5', 'item5', 4.0),
        ('user5', 'item6', 5.0),
    ]
    
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
    
    return {
        'interactions_df': df,
        'user_ids': sorted(df['user_id'].unique()),
        'item_ids': sorted(df['item_id'].unique())
    }

def test_recommender_initialization():
    """Test recommender initialization with default config."""
    recommender = AdvancedHybridRecommender()
    assert recommender.config.method == 'hybrid'
    assert recommender.config.weights['tensorflow'] == 0.6
    assert recommender.config.weights['implicit'] == 0.4

def test_recommender_custom_config():
    """Test recommender initialization with custom config."""
    config = RecommenderConfig(
        method='tensorflow',
        weights={'tensorflow': 1.0, 'implicit': 0.0},
        tensorflow_params={
            'embedding_dim': 32,
            'learning_rate': 0.002,
            'batch_size': 64,
            'epochs': 10,
            'validation_split': 0.1
        }
    )
    recommender = AdvancedHybridRecommender(config)
    assert recommender.config.method == 'tensorflow'
    assert recommender.config.weights['tensorflow'] == 1.0
    assert recommender.config.tensorflow_params['embedding_dim'] == 32

@pytest.mark.skip(reason="TensorFlow training can be slow in tests")
def test_recommender_fit(sample_data):
    """Test fitting the recommender."""
    recommender = AdvancedHybridRecommender(RecommenderConfig(
        tensorflow_params={'epochs': 1}  # Fast training for tests
    ))
    recommender.fit(
        interactions_df=sample_data['interactions_df']
    )
    assert len(recommender.user_mapping) == len(sample_data['user_ids'])
    assert len(recommender.item_mapping) == len(sample_data['item_ids'])
    
    # Test model initialization
    assert recommender.tf_model is not None
    assert isinstance(recommender.tf_model, tf.keras.Model)
    assert recommender.implicit_model is not None

def test_recommend_items(sample_data):
    """Test item recommendations."""
    recommender = AdvancedHybridRecommender(RecommenderConfig(
        method='implicit'  # Use only implicit for faster testing
    ))
    recommender.fit(
        interactions_df=sample_data['interactions_df']
    )
    
    recommendations = recommender.recommend_items('user1', n_recommendations=3)
    assert len(recommendations) == 3
    assert all(item in sample_data['item_ids'] for item in recommendations)

def test_similar_items(sample_data):
    """Test finding similar items."""
    recommender = AdvancedHybridRecommender(RecommenderConfig(
        method='implicit'  # Use only implicit for faster testing
    ))
    recommender.fit(
        interactions_df=sample_data['interactions_df']
    )
    
    similar_items = recommender.get_similar_items('item1', n_similar=2)
    assert len(similar_items) == 2
    assert all(item in sample_data['item_ids'] for item in similar_items)

def test_exclude_rated_items(sample_data):
    """Test that rated items are excluded from recommendations."""
    recommender = AdvancedHybridRecommender(RecommenderConfig(
        method='implicit'  # Use only implicit for faster testing
    ))
    recommender.fit(
        interactions_df=sample_data['interactions_df']
    )
    
    user_id = 'user1'
    recommendations = recommender.recommend_items(user_id, exclude_rated=True)
    
    # Get rated items for user1
    rated_items = set(
        sample_data['interactions_df'][
            sample_data['interactions_df']['user_id'] == user_id
        ]['item_id']
    )
    
    assert all(item not in rated_items for item in recommendations)

def test_tensorflow_model_architecture():
    """Test TensorFlow model architecture."""
    recommender = AdvancedHybridRecommender()
    
    # Initialize model
    tf_model = recommender.tf_model = recommender.tf_model = TensorFlowRecommender(
        num_users=100,
        num_items=50,
        embedding_dim=32
    )
    
    # Test model layers
    assert len(tf_model.dense_layers) == 3
    assert isinstance(tf_model.user_embedding, tf.keras.layers.Embedding)
    assert isinstance(tf_model.item_embedding, tf.keras.layers.Embedding)
    
    # Test embedding dimensions
    assert tf_model.user_embedding.output_dim == 32
    assert tf_model.item_embedding.output_dim == 32
    
    # Test forward pass
    batch_size = 16
    user_input = tf.zeros((batch_size,), dtype=tf.int32)
    item_input = tf.zeros((batch_size,), dtype=tf.int32)
    output = tf_model((user_input, item_input))
    
    assert output.shape == (batch_size, 1)

def test_data_preprocessing(sample_data):
    """Test data preprocessing and scaling."""
    recommender = AdvancedHybridRecommender()
    
    # Fit the recommender
    recommender.fit(
        interactions_df=sample_data['interactions_df']
    )
    
    # Test scaling
    assert hasattr(recommender, 'scaler')
    assert recommender.scaler is not None
    
    # Test that ratings are properly scaled
    original_ratings = sample_data['interactions_df']['rating'].values
    scaled_ratings = recommender.scaler.transform(original_ratings.reshape(-1, 1)).ravel()
    
    assert scaled_ratings.mean() == pytest.approx(0, abs=1e-7)
    assert scaled_ratings.std() == pytest.approx(1, abs=1e-7) 