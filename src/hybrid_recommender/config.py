"""Configuration for the hybrid recommender system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, TypedDict

class TensorFlowParams(TypedDict):
    """Parameters for the TensorFlow model."""
    embedding_dim: int
    learning_rate: float
    batch_size: int
    epochs: int
    validation_split: float

class ImplicitParams(TypedDict):
    """Parameters for the Implicit model."""
    factors: int
    regularization: float
    alpha: float
    iterations: int

@dataclass
class RecommenderConfig:
    """Configuration for the hybrid recommender system.
    
    Attributes:
        method: The recommendation method to use
        weights: Weights for different models in the ensemble
        tensorflow_params: Parameters for the TensorFlow model
        implicit_params: Parameters for the Implicit ALS model
    """
    
    method: Literal["hybrid", "tensorflow", "implicit"] = "hybrid"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "tensorflow": 0.6,
        "implicit": 0.4
    })
    
    # TensorFlow parameters
    tensorflow_params: Dict[str, float] = field(default_factory=lambda: {
        "embedding_dim": 50,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "validation_split": 0.2
    })
    
    # Implicit parameters
    implicit_params: Dict[str, float] = field(default_factory=lambda: {
        "factors": 50,
        "regularization": 0.01,
        "alpha": 40,
        "iterations": 30
    })
    
    def __post_init__(self):
        """Set default parameters if none provided."""
        self.weights = self.weights or {
            "tensorflow": 0.6,
            "implicit": 0.4
        }
        
        self.tensorflow_params = self.tensorflow_params or {
            "embedding_dim": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 20,
            "validation_split": 0.2
        }
        
        self.implicit_params = self.implicit_params or {
            "factors": 50,
            "regularization": 0.01,
            "alpha": 40,
            "iterations": 30
        } 