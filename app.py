'''
transformers	Loads pretrained AI models
torch	Runs deep learning models
sentencepiece	Tokenizer used by MarianMT
streamlit	Creates a web interface
'''
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model
tokenizer,model = load_model()

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
    with st.spinner("Translating.."):
        result = translate(text)
    st.success(result)