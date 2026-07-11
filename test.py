from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "good morning"

inputs = tokenizer(text, return_tensors="pt")
generated = model.generate(**inputs)

print("Model:", model.config._name_or_path)
print("Translation:", tokenizer.decode(generated[0], skip_special_tokens=True))