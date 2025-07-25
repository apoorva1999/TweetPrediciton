import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import f1_score
from tqdm import tqdm
from src.prompt import PROMPT_TEMPLATE

# Load model and tokenizer from Hugging Face Hub
model = T5ForConditionalGeneration.from_pretrained("apoooooorva/flan-t5-large-tweet-classification-v3")
tokenizer = T5Tokenizer.from_pretrained("apoooooorva/flan-t5-large-tweet-classification-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load dataset
df = pd.read_csv("./data/Q2_20230202_majority.csv").dropna(subset=["tweet", "label_true"])

# Prepare prompts using PROMPT_TEMPLATE
prompts = df["tweet"].apply(lambda t: PROMPT_TEMPLATE.format(tweet=t)).tolist()

# Generate predictions
label_pred = []
for prompt in tqdm(prompts, desc="Generating predictions"):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
    pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    label_pred.append(pred)

# Add predictions to DataFrame and save
df["label_pred"] = label_pred
df.to_csv("./results/Q2_20230202_majority.csv", index=False)

# Calculate F1 score
f1 = f1_score(df["label_true"], df["label_pred"], average="weighted")
print(f"F1 Score: {f1:.4f}")
