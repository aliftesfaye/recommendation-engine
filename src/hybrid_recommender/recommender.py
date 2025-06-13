"""Main recommender system implementation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from implicit.als import AlternatingLeastSquares
from typing import List, Dict, Optional, Union, Tuple, Any

from .config import RecommenderConfig

class TensorFlowRecommender(tf.keras.Model):
    """Neural network-based recommender model."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_dim, name="user_embedding"
        )
        self.item_embedding = tf.keras.layers.Embedding(
            num_items, embedding_dim, name="item_embedding"
        )
        self.dense_layers = [
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)
        ]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        user_input, item_input = inputs
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # Concatenate user and item embeddings
        x = tf.concat([user_embedded, item_embedded], axis=1)
        
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x)
            
        return x

class AdvancedHybridRecommender:
    """A hybrid recommendation system combining deep learning and implicit feedback approaches."""

    def __init__(self, config: Optional[RecommenderConfig] = None):
        """
        Initialize the hybrid recommender system.
        
        Args:
            config: Configuration for the recommender system
        """
        self.config = config or RecommenderConfig()
        
        # Initialize models
        self.tf_model: Optional[TensorFlowRecommender] = None
        self.implicit_model = AlternatingLeastSquares(**self.config.implicit_params)
        
        # Initialize storage
        self.user_mapping: Dict[Union[str, int], int] = {}
        self.item_mapping: Dict[Union[str, int], int] = {}
        self.reverse_user_mapping: Dict[int, Union[str, int]] = {}
        self.reverse_item_mapping: Dict[int, Union[str, int]] = {}
        self.scaler = StandardScaler()
        
    def _create_mappings(self, user_ids: List[Any], item_ids: List[Any]) -> None:
        """Create user and item ID mappings.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        self.user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_user_mapping = {idx: uid for uid, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: iid for iid, idx in self.item_mapping.items()}

    def fit(self, 
            interactions_df: pd.DataFrame,
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            rating_col: str = 'rating') -> None:
        """
        Fit the hybrid recommender system.
        
        Args:
            interactions_df: DataFrame containing user-item interactions
            user_col: Name of the user ID column
            item_col: Name of the item ID column
            rating_col: Name of the rating column
        """
        # Create mappings
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()
        self._create_mappings(unique_users, unique_items)
        
        # Create sparse matrix for implicit model
        user_idx = interactions_df[user_col].map(self.user_mapping)
        item_idx = interactions_df[item_col].map(self.item_mapping)
        ratings = interactions_df[rating_col].values
        
        # Scale ratings
        scaled_ratings = self.scaler.fit_transform(ratings.reshape(-1, 1)).ravel()
        
        # Create sparse matrix
        sparse_matrix = tf.sparse.SparseTensor(
            indices=tf.stack([user_idx, item_idx], axis=1),
            values=scaled_ratings,
            dense_shape=[len(self.user_mapping), len(self.item_mapping)]
        )
        
        if self.config.method in ['hybrid', 'tensorflow']:
            self._fit_tensorflow(sparse_matrix)
            
        if self.config.method in ['hybrid', 'implicit']:
            self._fit_implicit(sparse_matrix)

    def _fit_tensorflow(self, sparse_matrix: tf.sparse.SparseTensor) -> None:
        """Fit TensorFlow model."""
        # Initialize model if not already done
        if self.tf_model is None:
            self.tf_model = TensorFlowRecommender(
                num_users=len(self.user_mapping),
                num_items=len(self.item_mapping),
                embedding_dim=self.config.tensorflow_params['embedding_dim']
            )
        
        # Prepare training data
        indices = sparse_matrix.indices.numpy()
        users = indices[:, 0]
        items = indices[:, 1]
        ratings = sparse_matrix.values.numpy()
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            ((users, items), ratings)
        ).shuffle(10000).batch(self.config.tensorflow_params['batch_size'])
        
        # Compile and train
        self.tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.tensorflow_params['learning_rate']
            ),
            loss='mse'
        )
        
        self.tf_model.fit(
            dataset,
            epochs=self.config.tensorflow_params['epochs'],
            validation_split=self.config.tensorflow_params['validation_split']
        )

    def _fit_implicit(self, sparse_matrix: tf.sparse.SparseTensor) -> None:
        """Fit Implicit ALS model."""
        # Convert TensorFlow sparse tensor to scipy sparse matrix
        csr_matrix = tf.sparse.to_coo(sparse_matrix).to_scipy_sparse().tocsr()
        self.implicit_model.fit(csr_matrix.T)

    def recommend_items(self, 
                       user_id: Union[str, int],
                       n_recommendations: int = 5,
                       exclude_rated: bool = True) -> List[Union[str, int]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of recommended item IDs
        """
        user_idx = self.user_mapping[user_id]
        scores = np.zeros(len(self.item_mapping))
            
        if self.config.method in ['hybrid', 'tensorflow']:
            scores += self._get_tensorflow_scores(user_idx)
            
        if self.config.method in ['hybrid', 'implicit']:
            scores += self._get_implicit_scores(user_idx)
            
        if exclude_rated:
            # Get user's rated items from implicit model's internal representation
            rated_items = self.implicit_model.user_factors[user_idx].nonzero()[0]
            scores[rated_items] = float('-inf')
            
        top_items_idx = np.argsort(scores)[::-1][:n_recommendations]
        return [self.reverse_item_mapping[idx] for idx in top_items_idx]

    def _get_tensorflow_scores(self, user_idx: int) -> np.ndarray:
        """Get scores from TensorFlow model."""
        # Create input tensors for all items
        user_input = np.full(len(self.item_mapping), user_idx)
        item_input = np.arange(len(self.item_mapping))
        
        # Get predictions
        predictions = self.tf_model.predict(
            [user_input, item_input],
            batch_size=self.config.tensorflow_params['batch_size']
        )
        
        return self.config.weights['tensorflow'] * predictions.ravel()

    def _get_implicit_scores(self, user_idx: int) -> np.ndarray:
        """Get scores from Implicit model."""
        # Get user factors
        user_factors = self.implicit_model.user_factors[user_idx]
        # Get scores for all items
        scores = user_factors.dot(self.implicit_model.item_factors.T)
        
        return self.config.weights['implicit'] * scores

    def get_similar_items(self, 
                         item_id: Union[str, int], 
                         n_similar: int = 5) -> List[Union[str, int]]:
        """
        Find similar items using both models.
        
        Args:
            item_id: ID of the item
            n_similar: Number of similar items to return
            
        Returns:
            List of similar item IDs
        """
        item_idx = self.item_mapping[item_id]
        
        # Get similar items from implicit model
        implicit_similar = self.implicit_model.similar_items(item_idx, n_similar + 1)[0]
        implicit_similar = implicit_similar[implicit_similar != item_idx][:n_similar]
        
        # Get similar items from TensorFlow model's item embeddings
        if self.config.method in ['hybrid', 'tensorflow']:
            tf_embeddings = self.tf_model.item_embedding.embeddings
            query_embedding = tf_embeddings[item_idx]
            similarities = tf.matmul(
                tf_embeddings,
                tf.reshape(query_embedding, [-1, 1])
            ).numpy().ravel()
            
            # Exclude the query item
            similarities[item_idx] = float('-inf')
            tf_similar = np.argsort(similarities)[::-1][:n_similar]
            
            # Combine results using weights
            combined_items = set(implicit_similar) | set(tf_similar)
            scores = np.zeros(len(self.item_mapping))
            
            for idx in combined_items:
                if idx in implicit_similar:
                    scores[idx] += self.config.weights['implicit']
                if idx in tf_similar:
                    scores[idx] += self.config.weights['tensorflow']
                    
            similar_items = np.argsort(scores)[::-1][:n_similar]
        else:
            similar_items = implicit_similar
            
        return [self.reverse_item_mapping[idx] for idx in similar_items] 