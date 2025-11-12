import requests
import json

# === Configuration ===
EURI_API_URL = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_API_KEY = "your-api-key"  # Replace with your actual API key
MODEL = "gpt-4.1-nano"

# === Prompt Template for Key Information Extraction ===
def build_prompt(text):
    return f"""
You are an intelligent information extractor. Extract the key entities from the following text and return them in structured JSON format.

Text:
\"\"\"{text}\"\"\"

Extract the following fields if present:
- Full Name
- Email
- Phone Number
- Date
- Address
- Company
- Designation
- Skills
- Education
- Experience
- Any other useful metadata

Respond in clean JSON format only.
"""

# === Function to Extract Info ===
def extract_info(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional key information extractor."},
            {"role": "user", "content": build_prompt(text)}
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    }

    response = requests.post(EURI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        try:
            result = data['choices'][0]['message']['content']
            print("📦 Extracted Information:")
            print(result)
        except Exception as e:
            print("❌ Error extracting response:", e)
    else:
        print(f"❌ Request failed with status {response.status_code}")
        print(response.text)


# === Example Input ===
if __name__ == "__main__":
    unstructured_text = """
    Hello, my name is Rahul Sharma. I completed my B.Tech in Computer Science from IIT Delhi in 2019.
    I have 4 years of experience working at Infosys as a Software Engineer. You can contact me at rahul.sharma@gmail.com
    or call me at +91-9876543210. I specialize in Python, Flask, and NLP-based projects.
    """

    extract_info(unstructured_text)
