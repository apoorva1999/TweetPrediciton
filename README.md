# Tweet Stance Classification using Flan-T5-Large

This project implements a tweet stance classification system that fine-tunes a T5 model to classify tweets as **'in-favor'**, **'against'**, or **'neutral-or-unclear'** regarding COVID-19 vaccines.

## Features

- Fine-tunes a T5 (Flan-T5-Large) model for stance classification.
- Customizable prompt templates.
- Weighted F1 score evaluation.
- Modular and extensible codebase.

## File Structure

```
TweetPrediciton/
├── src/
│   ├── fine_tuning.py      # Main training and evaluation pipeline
│   └── prompt.py           # Prompt template for T5 input
├── Q2_20230202_majority.csv # Example dataset (CSV)
├── results/                # Output directory for trained models and results
└── README.md               # Project documentation
```
## Fine-tuned Model

The latest fine-tuned model is available on Hugging Face Hub:  
[apoooooorva/flan-t5-large-tweet-classification-v3](https://huggingface.co/apoooooorva/flan-t5-large-tweet-classification-v3)

- You can use this model for inference directly in your code or scripts.
- The model is compatible with Hugging Face `transformers` and can be loaded with:
  ```python
  from transformers import T5Tokenizer, T5ForConditionalGeneration
  tokenizer = T5Tokenizer.from_pretrained("apoooooorva/flan-t5-large-tweet-classification-v3")
  model = T5ForConditionalGeneration.from_pretrained("apoooooorva/flan-t5-large-tweet-classification-v3")
  ```


## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd TweetPrediciton
   ```

2. **Install dependencies**  
   **⚠️ Use Python version 3.11**
   It is recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**  
   - Place your CSV file (e.g., `Q2_20230202_majority.csv`) in the project root.
   - The CSV should have at least two columns: `tweet` and `label_true`.

4. **Edit the prompt template (optional)**  
   - Modify `src/prompt.py` to change how tweets are formatted for the model.

5. **Run training**  
   ```bash
   python src/fine_tuning.py
   ```

   The script will train the model and save results in the `results/` directory.
6. **Run Inference** 
This will load the fine-tuned model, run inference on the dataset, calculate f1 score and generate csv
   ```bash
   python src/infer_and_score.py
   ```
## Inference

You can use the provided script to run inference with the fine-tuned model and evaluate its performance:

1. **Load the fine-tuned model from Hugging Face Hub**
2. **Generate predictions for each tweet in the dataset**
3. **Calculate the weighted F1 score**
4. **Save a new CSV with predictions**

Run the following command:
```bash
python src/infer_and_score.py
```

- The script will automatically download the model from [Hugging Face](https://huggingface.co/apoooooorva/flan-t5-large-tweet-classification-v3).
- It will process `Q2_20230202_majority.csv` and output `Q2_20230202_majority_with_preds.csv` with an additional `label_pred` column.
- The weighted F1 score will be printed to the console.

### Example CSV Format

| tweet                                | label_true           | label_pred           |
|--------------------------------------|----------------------|----------------------|
| Vaccines are safe and effective.     | in-favor             | in-favor             |
| I don't trust the vaccine.           | against              | against              |
| Not sure about the vaccine's effects | neutral-or-unclear   | neutral-or-unclear   |

## Customization

- Adjust hyperparameters (batch size, epochs, etc.) in the `train()` call.


![Alt text](/image_for_fun.png)


