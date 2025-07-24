"""
Metrics Module

This module implements the MetricsCalculator class for computing evaluation
metrics for the tweet stance classification task.
"""

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch


class MetricsCalculator:
    """Class for computing evaluation metrics."""
    
    def __init__(self, tokenizer):
        """
        Initialize the metrics calculator.
        
        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics from model predictions.
        
        Args:
            eval_pred (tuple): (predictions, labels) from model evaluation
            
        Returns:
            dict: Dictionary with metric values
        """
        predictions, labels = eval_pred
        
        # Handle tuple predictions (from encoder-decoder models)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert logits to token IDs if necessary
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Decode predictions to text
        pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 (padding token) with tokenizer's pad_token_id
        labels = np.where(np.array(labels) != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode labels to text
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up whitespace
        pred_str = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        print("Predictions:", pred_str)
        print("Labels:", label_str)
        
        # Compute F1 score (macro average for multi-class)
        f1 = {"f1": f1_score(label_str, pred_str, average="macro")}
        print("F1 Score:", f1)
        return f1
    
    