import os
import json
import joblib
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config
from sklearn.preprocessing import LabelEncoder

# Custom multi-task model definition
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

# Load custom configuration for GPT-2
custom_config = GPT2Config(
    n_layer=4,           # Number of layers
    n_head=4,            # Number of attention heads
    n_embd=512           # Embedding size
)

# Define number of classes for each task
NUM_CLASSES = {
    'emotion': 0,  # Placeholder
    'sentiment': 0,  # Placeholder
    'intent': 0,  # Placeholder
    'toxicity': 0,  # Placeholder
    'spam': 0,  # Placeholder
    'sarcasm': 0  # Placeholder
}

# Load label encoders
label_encoders = {}
for task in NUM_CLASSES:
    label_encoders[task] = joblib.load(os.path.join('../models', f'{task}_label_encoder.pkl'))
    NUM_CLASSES[task] = len(label_encoders[task].classes_)

# Create the multi-task model with the custom configuration
model = TFGPT2ForMultiTask(custom_config, NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the weights
model.load_weights('../models/multi_task_model.weights.h5')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def preprocess_comment(comment, tokenizer, max_length):
    tokenized_text = tokenizer.encode(comment)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences([tokenized_text], maxlen=max_length, padding='post')
    return np.array(padded_text)

def analyze_comment(comment):
    max_length = 100
    preprocessed_comment = preprocess_comment(comment, tokenizer, max_length)
    predictions = model.predict(preprocessed_comment)
    
    decoded_predictions = {}
    for task in predictions:
        decoded_predictions[task] = label_encoders[task].inverse_transform(np.argmax(predictions[task], axis=1))[0]
    
    return decoded_predictions

def feedback(comment, sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback):
    feedback_choice = {
        "comment": comment,
        "sentiment": sentiment_feedback,
        "emotion": emotion_feedback,
        "intent": intent_feedback,
        "toxicity": toxicity_feedback,
        "sarcasm": sarcasm_feedback,
        "spam": spam_feedback
    }

    # Determine if any feedback choice is "No"
    if any(value == "No" for value in feedback_choice.values()):
        filename = 'feedback_with_no.json'
    else:
        filename = 'feedback_with_yes.json'

    # Append the feedback data to the appropriate JSON file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = []

    data.append(feedback_choice)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
