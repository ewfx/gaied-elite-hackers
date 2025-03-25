import os
from dotenv import load_dotenv
from openai import OpenAI
import email
from email.policy import default
import json


load_dotenv()

OPEN_API_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def classify_email(content, categories):
    response = client.responses.create(
        model="gpt-4o",
        instructions="Analyze the following email content and classify it into a Category from the provide category list as\n\n{categories}\n\nReturn only the sub-category.",
        input=content,
    )
    return response.output_text

# Function to extract email content


def extract_email_content(email_path):
    with open(email_path, "r", encoding="utf-8") as file:
        msg = email.message_from_file(file, policy=default)
        subject = msg["subject"]
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(
                        "utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(
                "utf-8", errors="ignore")
    return subject, body


# reading config
config = {}
with open('config.json') as json_data:
    config = json.load(json_data)

categories = list(config.keys())

# Process emails
EMAIL_DIR = "emails/"
data = []
for filename in os.listdir(EMAIL_DIR):
    if filename.endswith(".eml"):  # Adjust extension if needed
        subject, body = extract_email_content(
            os.path.join(EMAIL_DIR, filename))
        request_type, sub_type = classify_email(body, categories)
        data.append([subject, request_type, sub_type])
