import random
from llama3 import text_generation  # Import the synchronous function from llama3
from transformers import pipeline

# Initialize the sentiment analysis pipeline once
sentiment_pipeline = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")

def get_response(request: str) -> str:
    try:
        response = text_generation(request[1:])  # Call the synchronous function directly
        print(f"Generated response: {response}")
        return response
    except Exception as e:
        print(f"Error in text generation: {e}")
        return "I'm sorry, I'm having trouble processing your request. Please try again later."

def check_message(request: str) -> bool:
    classification = sentiment_pipeline(request)
    print(f"Classification: {classification}")
    if classification[0]['label'].upper() == 'NEG':
        return True
    else:
        return False