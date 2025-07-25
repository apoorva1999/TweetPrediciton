"""
Data Loading and Preprocessing for Tweet Stance Classification

This module implements the DataProcessor and TweetDataset classes used for
loading, preprocessing, and preparing tweet dataset to fed into the model for fine-tuning
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from prompt import PROMPT_TEMPLATE


class DataProcessor:
    """Class for loading and preprocessing tweet data."""
    
    def __init__(self, data_path):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the CSV file containing tweet data
        """
        self.data_path = data_path
    
    def load_data(self):
        """
        Load tweet data from CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing tweets and labels
        """
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["tweet", "label_true"])
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and validation sets.
        
        Args:
            df (pd.DataFrame): DataFrame to split
            test_size (float): Proportion of data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (training_df, validation_df)
        """
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df, val_df


class TweetDataset(Dataset):
    """Dataset class for tweet stance classification."""
    
    def __init__(self, dataframe, tokenizer, max_length=256):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with tweets and labels
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx (int): Index of the example
            
        Returns:
            dict: Dictionary with input_ids, attention_mask, and labels
        """
        # Extract tweet and label from dataframe
        tweet = self.data.iloc[idx]["tweet"]
        label = self.data.iloc[idx]["label_true"]
        
        # Format input and target texts
        input_text = PROMPT_TEMPLATE.format(tweet=tweet)
        target_text = label
        
        # Tokenize input
        input_enc = self.tokenizer(
            input_text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Tokenize target
        target_enc = self.tokenizer(
            target_text, 
            truncation=True, 
            padding="max_length", 
            max_length=10, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": target_enc.input_ids.squeeze(),
        }