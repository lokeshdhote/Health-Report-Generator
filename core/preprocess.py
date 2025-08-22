import re

def clean_text(text: str) -> str:
    """
    Cleans the input text for NLP processing:
    - Converts text to lowercase
    - Removes characters except letters, numbers, spaces, periods, and commas
    - Strips extra spaces
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove unwanted characters, keep letters, numbers, spaces, periods, commas
    text = re.sub(r'[^a-z0-9\s\.,]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
