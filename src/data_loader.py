import pandas as pd
import time
from tqdm import tqdm
import os
from predictor import Predictor
from sklearn.metrics import f1_score


def load_and_process_data(csv_path, model_path, sample_size=None):
    """     
    Load CSV data and process it with the predictor
    
    Args:
        csv_path (str): Path to the CSV file
        model_path (str): Path to the model
        sample_size (int, optional): Number of samples to process (for testing)
    
    Returns:
        pd.DataFrame: DataFrame with original data and predictions
    """
    
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Take a sample if specified
    if sample_size:
        df = df.head(sample_size)

    # Initialize the predictor
    predictor = Predictor(model_path)
    
    # Process tweets and get predictions
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tweets"):
        try:
            tweet = row['tweet']
            prediction = predictor.predict(tweet)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing tweet {idx}: {e}")
            predictions.append("error")
    
    # Add predictions to the dataframe
    df['prediction'] = predictions
    
    return df

def analyze_results(df):
    """
    Analyze the prediction results
    
    Args:
        df (pd.DataFrame): DataFrame with predictions
    """
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    # Show some example predictions
    print("\nExample predictions:")
    for idx, row in df.head(10).iterrows():
        print(f"Tweet: {row['tweet'][:100]}...")
        print(f"True label: {row['label_true']}")
        print(f"Predicted: {row['prediction']}")
        print("-" * 50)
    
    # Count predictions
    prediction_counts = df['prediction'].value_counts()
    print(f"\nPrediction distribution:")
    print(prediction_counts)
    
    # Compare with true labels (if available)
    if 'label_true' in df.columns:
        print(f"\nTrue label distribution:")
        print(df['label_true'].value_counts())
        
        # Calculate accuracy for non-error predictions
        valid_predictions = df[df['prediction'] != 'error']
        if len(valid_predictions) > 0:
            accuracy = (valid_predictions['prediction'] == valid_predictions['label_true']).mean()
            print(f"\nAccuracy: {accuracy:.2%}")

        # Calculate F1 scores for each class
        # Get unique classes from both true labels and predictions
        labels = ['in-favor', 'against', 'neutral-or-unclear']
        
        # Calculate F1 scores for each class
        f1_scores = f1_score(df['label_true'], df['prediction'], average=None, labels=labels)
        f1_scores_with_labels = {label:score for label,score in zip(labels, f1_scores)}
        print(f"\nF1 Scores by Class:")
        print(f1_scores_with_labels)
    
        # Also calculate overall F1 score (weighted average)
        f1_weighted = f1_score(df['label_true'], df['prediction'], average='weighted')
        print(f"Overall F1 Score (Weighted): {f1_weighted:.3f}")
    
    return df

def save_results(df, output_path):
    """
    Save results to CSV
    
    Args:
    df (pd.DataFrame): DataFrame with predictions
        output_path (str): Path to save the results
    """
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

def main():
    """Main function to run the data processing pipeline"""
    
    # Configuration
    CSV_PATH = "data/Q2_20230202_majority.csv"
    MODEL_PATH = "google/flan-t5-large"
    OUTPUT_PATH = "results/predictions.csv"
    SAMPLE_SIZE = 50  # Set to None to process all data
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files in data directory:")
        if os.path.exists("data"):
            print(os.listdir("data"))
        else:
            print("data directory does not exist")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Process the data
    start_time = time.time()
    
    try:
        df = load_and_process_data(CSV_PATH, MODEL_PATH, 100)
        df = analyze_results(df)
        save_results(df, OUTPUT_PATH)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
