import sys
import os
import joblib
import gradio as gr
import json

# Add the root directory of the project to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_analysis1 import analyze_comment, feedback

def classify_and_get_feedback(comment):
    predictions = analyze_comment(comment)
    return predictions

def handle_feedback(comment, sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback):
    feedback(comment, sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback)
    return "Thank you for your feedback!"

with gr.Blocks() as demo:
    gr.Markdown("## BMTE")
    
    # Comment Input
    comment_input = gr.Textbox(label="Enter YouTube Comment", lines=4)
    analyze_button = gr.Button("Analyze Comment")
    
    # Output
    output = gr.Json(label="Analysis Result")
    
    # Feedback Components
    sentiment_feedback = gr.Radio(label="Sentiment Correct?", choices=["Yes", "No"], visible=False)
    emotion_feedback = gr.Radio(label="Emotion Correct?", choices=["Yes", "No"], visible=False)
    intent_feedback = gr.Radio(label="Intent Correct?", choices=["Yes", "No"], visible=False)
    toxicity_feedback = gr.Radio(label="Toxicity Correct?", choices=["Yes", "No"], visible=False)
    sarcasm_feedback = gr.Radio(label="Sarcasm Correct?", choices=["Yes", "No"], visible=False)
    spam_feedback = gr.Radio(label="Spam Correct?", choices=["Yes", "No"], visible=False)
    
    submit_button = gr.Button("Submit Feedback", visible=False)
    
    feedback_status = gr.Textbox(label="Feedback Status", visible=False)
    
    # Functions
    def show_feedback_interface(predictions, comment):
        return {sentiment_feedback: gr.update(visible=True),
                emotion_feedback: gr.update(visible=True),
                intent_feedback: gr.update(visible=True),
                toxicity_feedback: gr.update(visible=True),
                sarcasm_feedback: gr.update(visible=True),
                spam_feedback: gr.update(visible=True),
                submit_button: gr.update(visible=True),
                feedback_status: gr.update(visible=False)}

    def submit_feedback(comment, sentiment, emotion, intent, toxicity, sarcasm, spam):
        feedback(comment, sentiment, emotion, intent, toxicity, sarcasm, spam)
        return {sentiment_feedback: gr.update(visible=False),
                emotion_feedback: gr.update(visible=False),
                intent_feedback: gr.update(visible=False),
                toxicity_feedback: gr.update(visible=False),
                sarcasm_feedback: gr.update(visible=False),
                spam_feedback: gr.update(visible=False),
                submit_button: gr.update(visible=False),
                feedback_status: gr.update(visible=True, value="Thank you for your feedback!")}

    # Event Binding
    analyze_button.click(classify_and_get_feedback, inputs=[comment_input], outputs=[output])
    analyze_button.click(show_feedback_interface, inputs=[output, comment_input], outputs=[sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback, submit_button, feedback_status])
    submit_button.click(submit_feedback, inputs=[comment_input, sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback], outputs=[sentiment_feedback, emotion_feedback, intent_feedback, toxicity_feedback, sarcasm_feedback, spam_feedback, submit_button, feedback_status])
    
demo.launch()
