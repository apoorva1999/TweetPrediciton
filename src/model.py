"""
Model Building Module

This module provides the ModelBuilder class for initializing and configuring
the T5 model for tweet stance classification.

Author: Apoorva Mittal
Date: July 24, 2025
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ModelBuilder:
    """Class for initializing and configuring the model."""
    
    def __init__(self, model_name="google/flan-t5-large"):
        """
        Initialize the model builder.
        
        Args:
            model_name (str): Name of the pre-trained model
        """
        self.model_name = model_name
    
    def build(self):
        """
        Build and initialize the model and tokenizer.
        
        Returns:
            tuple: (tokenizer, model, device)
        """
        # Initialize tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Model initialized on device: {device}")
        return tokenizer, model, device