"""
Tweet Stance Classification using T5 Models

This module implements a tweet stance classification system that fine-tunes
a T5 model to classify tweets as 'in-favor', 'against', or 'neutral-or-unclear'
regarding COVID-19 vaccines.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from prompt import PROMPT_TEMPLATE
from data import DataProcessor, TweetDataset
from model import ModelBuilder
from metrics import MetricsCalculator
from trainer import TrainerManager


class TweetClassifier:
    """Main class orchestrating the tweet classification process."""
    
    def __init__(self, data_path, model_name="google/flan-t5-large", output_dir="./results"):
        """
        Initialize the tweet classifier.
        
        Args:
            data_path (str): Path to dataset
            model_name (str): Name of pre-trained model
            output_dir (str): Output directory for results
        """
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize components
        self.data_processor = DataProcessor(data_path)
        self.trainer_manager = TrainerManager(output_dir)
    
    def train(self, test_size=0.2, max_length=256, 
              batch_size=2, epochs=5, learning_rate=3e-5):
        """
        Run the full training pipeline.
        
        Args:
            test_size (float): Proportion of data for validation
            max_length (int): Maximum input sequence length
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        # 1. Load and prepare data
        print("Loading and preparing data...")
        df = self.data_processor.load_data()
        train_df, val_df = self.data_processor.split_data(df, test_size=test_size)
        print(f"Data split: {len(train_df)} training, {len(val_df)} validation examples")
        
        # 2. Initialize model and tokenizer
        print(f"Initializing {self.model_name}...")
        model_builder = ModelBuilder(self.model_name)
        tokenizer, model, device = model_builder.build()
        
        # 3. Create datasets
        train_dataset = TweetDataset(train_df, tokenizer, max_length=max_length)
        val_dataset = TweetDataset(val_df, tokenizer, max_length=max_length)
        
        # 4. Initialize metrics calculator
        metrics_calculator = MetricsCalculator(tokenizer)
        
        # 5. Configure and create trainer
        training_args = self.trainer_manager.create_training_args(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        )
        
        trainer = self.trainer_manager.create_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics_fn=metrics_calculator.compute_metrics
        )
        
        # 6. Train model
        print("\nStarting training...")
        trainer.train()
        
        # 7. Evaluate with generate
        print("\nEvaluating with generate()...")
        f1, preds, refs = metrics_calculator.evaluate_with_generate(
            model, val_dataset, device
        )
        
        # 8. Save model
        print(f"\nSaving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        
        return model, {"f1": f1, "predictions": preds, "references": refs}


def main():
    """Main entry point for the application."""
    # Set current directory

    
    # Create classifier with configuration
    classifier = TweetClassifier(
        data_path="Q2_20230202_majority.csv",
        model_name="google/flan-t5-large", 
        output_dir="./results"
    )
    
    # Run training pipeline
    model, metrics = classifier.train(
        test_size=0.2,
        max_length=256,
        batch_size=2,
        epochs=5
    )
    
    print(f"\nTraining complete! Final F1 score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

