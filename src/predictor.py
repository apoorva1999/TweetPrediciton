from transformers import T5Tokenizer, T5ForConditionalGeneration
from prompt import PROMPT_TEMPLATE

class Predictor:
    def __init__(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def predict(self, tweet):
        input_text = PROMPT_TEMPLATE.format(tweet=tweet)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        input_ids = input_ids.to(next(self.model.parameters()).device)
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



