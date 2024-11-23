import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Instantiate the OpenAI client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def query_openai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Your task is to analyze experiment outputs rigorously. Provide detailed and structured "
                        "insights into performance metrics, focusing on accuracy, efficiency, and scalability. "
                        "Highlight patterns, anomalies, and key observations with examples. Avoid redundancy or "
                        "shallow statements—your analysis should be concise, clear, and critical, factual. Do not "
                        "hallucinate."

                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=3000,  # Adjust as needed
            temperature=0.2,
        )
        # Extract the assistant's reply
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error querying OpenAI: {str(e)}")
        return "Failed to get a response from OpenAI."
