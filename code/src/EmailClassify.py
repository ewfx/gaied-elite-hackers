!pip install pandas google-generativeai transformers scikit-learn


import os
import pandas as pd
import email
from email.policy import default
import google.generativeai as genai
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Set up API Keys
GEMINI_API_KEY = "your_key"
HUGGINGFACE_API_KEY = "your_key"

genai.configure(api_key=GEMINI_API_KEY)

# Load Hugging Face classifier (adjust model as needed)
classifier = pipeline(
    "text-classification", 
    model="facebook/bart-large-mnli", 
    token=HUGGINGFACE_API_KEY
)

# Initialize Gemini Client
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Directory containing email files
EMAIL_DIR = "emails/"

# Function to extract email content
def extract_email_content(email_path):
    with open(email_path, "r", encoding="utf-8") as file:
        msg = email.message_from_file(file, policy=default)
        subject = msg["subject"]
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    return subject, body

# Function to classify request type and subtype
def classify_email(content):
    # Step 1: Hugging Face Model for General Classification
    result = classifier(content)
    request_type = result[0]['label']

    # Step 2: Gemini for Sub-Type Extraction
    prompt = f"Analyze the following email content and classify it into a sub-category:\n\n{content}\n\nReturn only the sub-category."
    response = gemini_model.generate_content(prompt)
    sub_type = response.text.strip()

    return request_type, sub_type

# Process emails
data = []
for filename in os.listdir(EMAIL_DIR):
    if filename.endswith(".eml"):  # Adjust extension if needed
        subject, body = extract_email_content(os.path.join(EMAIL_DIR, filename))
        request_type, sub_type = classify_email(body)
        data.append([subject, request_type, sub_type])

# Convert to DataFrame and Save
df = pd.DataFrame(data, columns=["Email Subject", "Request Type", "Sub Request Type"])
df.to_csv("classified_emails.csv", index=False)

print("Classification completed. Results saved to 'classified_emails.csv'.")