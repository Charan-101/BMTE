import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from transformers import  GPT2Tokenizer, TFGPT2Model, GPT2Config
from tokenizers import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_data(data, tokenizer, max_length):
    texts = [item['content'] for item in data]
    emotions = [item['emotion'] for item in data]
    sentiments = [item['sentiment'] for item in data]
    intents = [item['intent'] for item in data]
    toxicities = [item['toxicity'] for item in data]
    sarcasms = [item['sarcasm'] for item in data]
    spams = [item['spam'] for item in data]

    # Tokenize texts
    tokenized_texts = [tokenizer.encode(text).ids for text in texts]
    padded_texts = pad_sequences(tokenized_texts, maxlen=max_length, padding='post')
    # Encode labels
    label_encoders = {}
    labels = {}

    for label, name in zip([emotions, sentiments, intents, toxicities, sarcasms, spams], ['emotion', 'sentiment', 'intent', 'toxicity', 'sarcasm', 'spam']):
        label_encoders[name] = LabelEncoder()
        encoded_labels = label_encoders[name].fit_transform(label)
        labels[name] = tf.keras.utils.to_categorical(encoded_labels)

    return padded_texts, labels, label_encoders

# Load custom tokenizer
tokenizer = Tokenizer.from_file("telugu_tokenizer_30k.json")

# Load data
data = load_data("../data/updated_spam.json")

# Define maximum length for padding
MAX_LENGTH = 100

# Preprocess data
X, labels, label_encoders = preprocess_data(data, tokenizer, MAX_LENGTH)

# Split the data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_train = {}
y_test = {}
for key in labels:
    y_train[key], y_test[key] = train_test_split(labels[key], test_size=0.2, random_state=42)

# Define the model
class TFGPT2ForMultiTask(tf.keras.Model):
    def __init__(self, config, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpt2 = TFGPT2Model(config)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.outputs = {}
        for task, num_class in num_classes.items():
            self.outputs[task] = tf.keras.layers.Dense(num_class, activation='softmax', name=task)

    def call(self, inputs, training=False):
        gpt2_outputs = self.gpt2(inputs)[0]
        pooled_output = gpt2_outputs[:, -1]  # Use the last token for classification
        pooled_output = self.dropout(pooled_output, training=training)
        
        task_outputs = {}
        for task in self.outputs:
            task_outputs[task] = self.outputs[task](pooled_output)
        
        return task_outputs

# Custom configuration for GPT-2
custom_config = GPT2Config(
    n_layer=4,           # Number of layers
    n_head=4,            # Number of attention heads
    n_embd=512           # Embedding size
)

NUM_CLASSES = {
    'emotion': len(label_encoders['emotion'].classes_),
    'sentiment': len(label_encoders['sentiment'].classes_),
    'intent': len(label_encoders['intent'].classes_),
    'toxicity': len(label_encoders['toxicity'].classes_),
    'spam': len(label_encoders['spam'].classes_),
    'sarcasm': len(label_encoders['sarcasm'].classes_)
}

# Create the multi-task model with the custom configuration
multi_task_model = TFGPT2ForMultiTask(custom_config, NUM_CLASSES)

# Compile the model with individual loss functions for each task
losses = {task: 'categorical_crossentropy' for task in NUM_CLASSES}
metrics = {task: ['accuracy'] for task in NUM_CLASSES}
multi_task_model.compile(optimizer='adam', loss=losses, metrics=metrics)

# Prepare the training and validation data for the multi-output model
y_train_list = {task: y_train[task] for task in NUM_CLASSES}
y_test_list = {task: y_test[task] for task in NUM_CLASSES}

# Define callbacks for model checkpointing and CSV logging
checkpoint_path = "../models/multi_task_model.weights.h5"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

csv_logger = CSVLogger('training_log.csv', append=True, separator=',')

# Train the model with checkpoints
history = multi_task_model.fit(
    X_train, y_train_list, epochs=15, batch_size=64, validation_split=0.2, callbacks=[checkpoint_callback, csv_logger]
)

# Save the label encoders
for task, encoder in label_encoders.items():
    joblib.dump(encoder, f'../models/{task}_label_encoder.pkl')

print("Models trained and saved successfully.")
