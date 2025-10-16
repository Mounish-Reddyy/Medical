import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# -----------------------------
# Title & Description
# -----------------------------
st.title("🩺 Medical Named Entity Recognition (NER)")
st.write("This app extracts **Diseases**, **Symptoms**, **Drugs**, and **Medical Tests** from medical text using NLP and rule-based methods.")

# -----------------------------
# Input Text
# -----------------------------
text_input = st.text_area("Enter medical text here:", 
                          "Patient was prescribed Paracetamol for fever and headache. Blood test was conducted at Apollo Hospital.")

# -----------------------------
# Define sample keyword lists
# -----------------------------
diseases = ["diabetes", "hypertension", "asthma", "cancer", "covid", "infection", "malaria", "fever"]
symptoms = ["cough", "headache", "pain", "vomiting", "cold", "fatigue", "nausea"]
drugs = ["paracetamol", "ibuprofen", "aspirin", "amoxicillin", "insulin", "azithromycin"]
tests = ["blood test", "x-ray", "mri", "ct scan", "urine test", "ecg"]

# -----------------------------
# Function to extract entities
# -----------------------------
def medical_ner(text):
    tokens = word_tokenize(text.lower())
    entities = {"Disease": [], "Symptom": [], "Drug": [], "Test": []}

    for i in range(len(tokens)):
        word = tokens[i]
        bigram = " ".join(tokens[i:i+2])

        if word in diseases:
            entities["Disease"].append(word)
        if word in symptoms:
            entities["Symptom"].append(word)
        if word in drugs:
            entities["Drug"].append(word)
        if word in tests or bigram in tests:
            entities["Test"].append(bigram if bigram in tests else word)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

# -----------------------------
# Run the extraction
# -----------------------------
if st.button("Extract Entities"):
    results = medical_ner(text_input)
    
    st.subheader("🔍 Extracted Entities:")
    if any(results.values()):
        for category, items in results.items():
            if items:
                st.write(f"**{category}:** {', '.join(items)}")
    else:
        st.warning("No medical entities found. Try a different text.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit + NLTK | Medical NER Project")

