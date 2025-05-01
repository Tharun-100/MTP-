# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

from modeling import binary_KL, attnmaskbin, getattnmask, MultiHeadAttention_wpos, create_model
from data_preprocessing import seq_to_embeds

# Define constants
windowsize = 145  
seqminsize = 149
aas = "ACDEFGHIKLMNPQRSTVWY"
MODEL_PATH = "canya_model.h5"

# Define custom objects for loading
custom_objects = {
    'binary_KL': binary_KL,
    'attnmaskbin': attnmaskbin,
    'getattnmask': getattnmask,
    'MultiHeadAttention_wpos': MultiHeadAttention_wpos
}

# Load trained model
@st.cache(allow_output_mutation=True)  
def load_trained_model(model_path=MODEL_PATH):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Preprocessing function
def preprocess_sequence(seq):
    """
    Clean and embed the sequence, pad if needed.
    """
    # Basic cleaning
    if "X" in seq or "Z" in seq:
        raise ValueError("Invalid amino acids (X or Z) detected.")
    
    seq = seq.split('*')[0]
    seq = "MDSTVSNFQKILTNPQNGGAQSNYGGGQSQGGYQQQNGGGQQQGGYQQQNYGPQGSWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIENF" + seq

    # Break into chunks if needed
    if len(seq) > windowsize:
        seq_chunks = [seq[j:j+windowsize] for j in range(0, len(seq), windowsize)]
    else:
        seq_chunks = [seq]

    embeddedseqs = []

    for chunk in seq_chunks:
        embed = seq_to_embeds(chunk)
        if embed.shape[0] < seqminsize:
            padding_needed = seqminsize - embed.shape[0]
            embed = np.pad(embed, ((0, padding_needed), (0, 0)), 'constant')
        embeddedseqs.append(embed)

    embeddedseqs = np.array(embeddedseqs)

    return embeddedseqs

# Prediction function
def predict_sequence(model, embeddedseqs):
    """
    Predict using the model and return majority class.
    """
    y_probs = model.predict(embeddedseqs)
    y_preds = (y_probs > 0.5).astype(int).flatten()
    majority_class = int(np.round(np.mean(y_preds)))

    return majority_class

# Streamlit App
def main():
    st.title("🧬 CANYA - Nucleation Propensity Prediction")

    st.write("""
    This tool predicts whether an input protein sequence has nucleation propensity using the trained CANYA model.
    """)

    # User input
    user_seq = st.text_area("Enter your protein sequence:", height=150)

    if st.button("Predict"):
        if user_seq.strip() == "":
            st.error("Please enter a valid protein sequence.")
        else:
            try:
                # Load model
                model = load_trained_model()

                # Preprocess input
                embeddedseqs = preprocess_sequence(user_seq.strip())

                # Predict
                label = predict_sequence(model, embeddedseqs)

                # Display result
                st.success(f"**Predicted Label:** {label}")

                if label == 1:
                    st.markdown("<h3 style='color: green;'>✅ This sequence has nucleation propensity!</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: red;'>❌ This sequence does NOT have nucleation propensity.</h3>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
