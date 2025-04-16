from openai import OpenAI
import os

# Initialize OpenAI client with the hardcoded API key
client = OpenAI(
    api_key="YOUR_OPENAI_API_KEY",
)

class FallbackResponse:
    """Fallback response when API fails"""
    def __init__(self, message_obj):
        self.choices = [type('obj', (object,), {'message': type('obj', (object,), message_obj)})]

def call_completion_api(model, messages):
    """OpenAI API call without caching"""
    # Ensure specific formatting instructions are included
    if isinstance(messages, list) and messages and 'content' in messages[0]:
        content = messages[0]['content']
        
        # Check if this is a probability calculation request
        if "probabilities" in content and "|" in content:
            # Add a clear instruction about formatting at the end of the prompt
            format_reminder = "\n\nIMPORTANT: Your response MUST follow the format specified above exactly, with each diagnosis-probability pair on a separate line using the pipe character (|) as separator."
            messages[0]['content'] = content + format_reminder
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800  # Limit response size for speed
        )
        return response
    except Exception as e:
        print(f"API error: {e}")
        # Create a simple fallback response
        message_obj = {"role": "assistant", "content": "Error occurred during API call"}
        return FallbackResponse(message_obj) 