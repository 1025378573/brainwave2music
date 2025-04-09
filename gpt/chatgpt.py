import openai
from gpt import api_keys


openai.api_key = api_keys.openai_key

def generate_text_from_keywords(keywords):
    prompt = f"Generate a long paragraph of text describing music based on the following keywords: {' '.join(keywords)}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates music descriptions."},
                {"role": "user", "content": prompt}
            ],
        )
        text = response.choices[0].message.content
        print(f"Generated Text: {text}")
        return text
    except Exception as e:
        print(f"Error interacting with OpenAI API: {e}")
        return "An error occurred while processing your request."