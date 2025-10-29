import os, requests, uuid, json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
region = os.environ["TRANSLATOR_TEXT_REGION"]
endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

path = "/detect?api-version=3.0"
constructed_url = endpoint + path

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Ocp-Apim-Subscription-Region": region,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}

# Test with multiple texts in different languages
body = [
    {"text": "Ahoj, jak se máš?"},  # Czech
    {"text": "Hello, how are you?"},  # English
    {"text": "Kolik je zaměstnanost v Praze?"},  # Czech
    {"text": "What is the employment rate in Prague?"},  # English
]

request = requests.post(constructed_url, headers=headers, json=body)
response = request.json()

print(
    json.dumps(
        response, sort_keys=True, indent=4, ensure_ascii=False, separators=(",", ": ")
    )
)

# Print simplified results
print("\n--- Simplified Results ---")
for i, item in enumerate(response):
    text = body[i]["text"]
    language = item["language"]
    score = item["score"]
    print(f"Text: '{text}' -> Language: {language} (confidence: {score})")
