# Tweet Stance Classification using T5

This project implements a tweet stance classification system that fine-tunes a T5 model to classify tweets as **'in-favor'**, **'against'**, or **'neutral-or-unclear'** regarding COVID-19 vaccines.

## Features

- Fine-tunes a T5 (e.g., Flan-T5) model for stance classification.
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

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd TweetPrediciton
   ```

2. **Install dependencies**  
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

## Example CSV Format

| tweet                                | label_true           |
|--------------------------------------|----------------------|
| Vaccines are safe and effective.     | in-favor             |
| I don't trust the vaccine.           | against              |
| Not sure about the vaccine's effects | neutral-or-unclear   |

## Customization

- Adjust hyperparameters (batch size, epochs, etc.) in the `train()` call.

