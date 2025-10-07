import os
import cv2
import base64
from openai import OpenAI
import re

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass

# Init GitHub Models client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)


def extract_result_from_message_prompt(raw_text):
    match = re.search(r"\*\*(.*?)\*\*", raw_text)
    if match:
        extracted_text = match.group(1).strip()
    else:
        # fallback: just strip everything after colon
        extracted_text = raw_text.split(":")[-1].strip()
    return extracted_text


def extract_text_from_image_llm(model_name: str, img_data_uri: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all readable text from this image region.",
                    },
                    {"type": "image_url", "image_url": {"url": img_data_uri}},
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


def extract_text_from_image_using_llm(region_of_interest, model_name: str = "gpt"):
    # Convert ROI (NumPy array) to base64
    _, buffer = cv2.imencode(".jpg", region_of_interest)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    img_data_uri = f"data:image/jpeg;base64,{img_base64}"
    if model_name == "mistral":
        model = "mistral-small-2503"
    elif model_name == "phi":
        model = "Phi-4-multimodal-instruct"
    else:
        model = "gpt-4o-mini"
    get_text_with_prompt = extract_text_from_image_llm(model, img_data_uri)
    get_number_plate_no = extract_result_from_message_prompt(get_text_with_prompt)
    return get_number_plate_no
