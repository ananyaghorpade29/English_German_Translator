'''
transformers	Loads pretrained AI models
torch	Runs deep learning models
sentencepiece	Tokenizer used by MarianMT
streamlit	Creates a web interface
'''
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate(text):
    inputs= tokenizer(
        text,
        return_tensors="pt",
        padding= True
    )
    translated= model.generate(**inputs)

    output = tokenizer.decode(
        translated[0],
        skip_special_tokens = True
    )
    print(output)
    return output

st.title("English to German Translator")
text= st.text_area("Enter English text")

if st.button("Translate"):
    result = translate(text)
    st.success(result)