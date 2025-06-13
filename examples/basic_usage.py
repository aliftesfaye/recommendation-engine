"""Basic usage example of the hybrid recommender system."""

import numpy as np
import pandas as pd
from typing import List
from hybrid_recommender import AdvancedHybridRecommender, RecommenderConfig

def main() -> None:
    # Create sample user-item interaction data
    user_item_matrix = np.array([
        [5, 3, 0, 1, 0, 2],  # User 1's ratings
        [4, 0, 0, 1, 2, 0],  # User 2's ratings
        [1, 2, 3, 4, 0, 0],  # User 3's ratings
        [0, 0, 4, 3, 5, 1],  # User 4's ratings
        [2, 1, 0, 0, 4, 5],  # User 5's ratings
    ], dtype=np.float32)

    # Create user and item IDs
    user_ids: List[str] = [f'user{i}' for i in range(1, 6)]
    item_ids: List[str] = [f'item{i}' for i in range(1, 7)]

    # Create a custom configuration
    config = RecommenderConfig(
        method='hybrid',
        weights={
            'tensorflow': 0.6,
            'implicit': 0.4
        },
        tensorflow_params={
            'embedding_dim': 50,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20,
            'validation_split': 0.2
        },
        implicit_params={
            'factors': 50,
            'regularization': 0.01,
            'alpha': 40,
            'iterations': 30
        }
    )

    # Initialize and train the recommender
    print("Initializing and training the recommender...")
    recommender = AdvancedHybridRecommender(config)
    
    # Convert to DataFrame for better handling of sparse data
    df = pd.DataFrame(user_item_matrix, index=user_ids, columns=item_ids)
    df = df.reset_index().melt(
        id_vars=['index'],
        var_name='item_id',
        value_name='rating'
    ).rename(columns={'index': 'user_id'})
    df = df[df['rating'] > 0]  # Remove non-interactions
    
    recommender.fit(
        interactions_df=df,
        user_col='user_id',
        item_col='item_id',
        rating_col='rating'
    )

    # Generate recommendations for each user
    print("\nGenerating recommendations for each user:")
    for user_id in user_ids:
        print(f"\nTop 3 recommendations for {user_id}:")
        recommendations = recommender.recommend_items(
            user_id=user_id,
            n_recommendations=3,
            exclude_rated=True
        )
        print(recommendations)

    # Find similar items
    print("\nFinding similar items:")
    for item_id in item_ids[:3]:
        print(f"\nItems similar to {item_id}:")
        similar_items = recommender.get_similar_items(
            item_id=item_id,
            n_similar=2
        )
        print(similar_items)

if __name__ == "__main__":
    main() 