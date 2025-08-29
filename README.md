# Next Word Prediction — LSTM Project
A Natural Language Processing (NLP) project that predicts the **next word in a sequence of text** using an LSTM (Long Short-Term Memory) neural network. The model is trained on the text of *Hamlet* and wrapped inside a Flask application for interactive predictions.
---
## Project Overview

This project demonstrates how to build, train, and deploy a **sequence prediction model** using deep learning.  
It includes:
- **Data Preprocessing** — preparing text data into sequences.  
- **Model Training**     — building and training an LSTM neural network.  
- **Model Saving**       — saving trained weights and tokenizer for later use.  
- **Flask App**          — serving predictions interactively via a simple web app.
    
---
## Repository Structure

Next_word_predicton_project/

├── app.py # Flask app for serving predictions
├── experiemnts.ipynb # Notebook for data prep, model training, evaluation
├── hamlet.txt # Training dataset (Shakespeare's Hamlet)
├── next_word_lstm.h5 # Trained LSTM model
├── tokenizer.pickle # Tokenizer used for text-to-sequence conversion
├── requirements.txt # Python dependencies
└── README.md # Documentation

---
## Getting Started
### Prerequisites
- Python 3.x  
- Jupyter Notebook (optional, for running experiments)  
- Dependencies listed in `requirements.txt`

### Installation

cmd
git clone https://github.com/Rohitcodermanit/Next_word_predicton_project.git
cd Next_word_predicton_project
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Usage

Training the Model
To experiment with training:
jupyter notebook experiemnts.ipynb
This walks through preprocessing, LSTM training, and evaluation.

Running Predictions

Run the Flask app:
python app.py
Open http://localhost:5000 in your browser, type a phrase, and the app will predict the most likely next word.

How It Works

Preprocessing

Input text (Hamlet) is tokenized.
Sequences are generated and padded for training.
Model Training
An LSTM network is trained on input sequences.
Loss minimized via categorical cross-entropy.

Prediction

Given a user prompt, tokenizer converts text into sequence.
Model (next_word_lstm.h5) outputs probabilities for the next word.
The predicted word is mapped back via tokenizer.

Model & Results

Architecture: Embedding layer + LSTM + Dense (Softmax).
Dataset: Shakespeare’s Hamlet.
Output: Predicts next probable word from a given input text.
Results and accuracy details available in experiemnts.ipynb.

Dependencies

Main libraries:
TensorFlow / Keras
Flask
NumPy, Pandas
Pickle (for tokenizer)
Check requirements.txt for exact versions.

Contributing

You can help by:
Adding new datasets for better generalization.
Improving the LSTM architecture (e.g., stacked LSTMs, GRUs, Transformers).
Extending the Flask app with a richer UI (Streamlit/Gradio).
