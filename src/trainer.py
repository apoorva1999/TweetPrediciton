"""
Trainer Manager for Tweet Stance Classification

This module implements the TrainerManager class, which is responsible for
initializing TrainingArguments and Trainer class required for training the model.
"""

from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from transformers import DataCollatorForSeq2Seq

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
                             epochs=5,
                             weight_decay=0.01,
                             logging_steps=10,
                             eval_strategy="epoch",
                             gradient_accumulation_steps=8,
                             label_smoothing_factor=0.1):
        """
        Create training arguments.
        
        Args:
            Various training hyperparameters
            
        Returns:
            TrainingArguments: Configuration for training
        """
        return Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            push_to_hub=False,
            report_to="none",
            eval_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=gradient_accumulation_steps,
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            label_smoothing_factor=label_smoothing_factor
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
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        return Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )