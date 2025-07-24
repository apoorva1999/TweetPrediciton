"""
Trainer Manager for Tweet Stance Classification

This module implements the TrainerManager class, which is responsible for
initializing TrainingArguments and Trainer class required for training the model.
"""

from transformers import Trainer, TrainingArguments


class TrainerManager:
    """Class for managing the training process."""
    
    def __init__(self, output_dir="./results"):
        """
        Initialize the trainer manager.
        
        Args:
            output_dir (str): Directory to save model and results
        """
        self.output_dir = output_dir
    
    def create_training_args(self, 
                             learning_rate=3e-5,
                             batch_size=2,
                             epochs=2,
                             weight_decay=0.01,
                             logging_steps=10,
                             eval_strategy="epoch",
                             ):
        """
        Create training arguments.
        
        Args:
            Various training hyperparameters
            
        Returns:
            TrainingArguments: Configuration for training
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            push_to_hub=False,
            report_to="none",
            evaluation_strategy=eval_strategy,
            save_strategy="epoch"
        )
    
    def create_trainer(self, model, args, train_dataset, val_dataset, 
                      tokenizer, compute_metrics_fn):
        """
        Create a Trainer instance.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            val_dataset: Validation dataset
            tokenizer: Tokenizer
            compute_metrics_fn: Function to compute metrics
            
        Returns:
            Trainer: Configured trainer
        """
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )