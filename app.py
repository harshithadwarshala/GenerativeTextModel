# =========================
# GENERATIVE TEXT MODEL
# Using LSTM & GPT (Transformers)
# =========================

# --- Install required libraries (if not already installed) ---
# !pip install tensorflow keras numpy transformers torch

# -------------------------
# 1. LSTM TEXT GENERATION
# -------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample training corpus
corpus = [
    "machine learning is fascinating",
    "deep learning is a subset of machine learning",
    "artificial intelligence is the future",
    "neural networks power deep learning",
    "text generation using lstm is interesting"
]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split predictors and label
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(total_words, activation='softmax'))

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X, y, epochs=200, verbose=0)

# Function to generate text with LSTM
def generate_text_lstm(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(lstm_model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

print("=== LSTM Generated Text ===")
print(generate_text_lstm("machine learning", 5))


# -------------------------
# 2. GPT TEXT GENERATION (Hugging Face)
# -------------------------

from transformers import pipeline

# Load GPT-2 small model for text generation
generator = pipeline("text-generation", model="gpt2")

print("\n=== GPT Generated Text ===")
prompt = "The future of artificial intelligence"
result = generator(prompt, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
print(result[0]['generated_text'])
