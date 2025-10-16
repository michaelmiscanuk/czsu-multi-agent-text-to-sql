import os, requests, uuid, json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

subscription_key = os.environ["TRANSLATOR_TEXT_SUBSCRIPTION_KEY"]
region = os.environ["TRANSLATOR_TEXT_REGION"]
endpoint = os.environ["TRANSLATOR_TEXT_ENDPOINT"]

path = "/translate?api-version=3.0"
params = "&to=en"
constructed_url = endpoint + path + params

headers = {
    "Ocp-Apim-Subscription-Key": subscription_key,
    "Ocp-Apim-Subscription-Region": region,
    "Content-type": "application/json",
    "X-ClientTraceId": str(uuid.uuid4()),
}

# You can pass more than one object in body.
body = [{"text": "Hello World!"}]
request = requests.post(constructed_url, headers=headers, json=body)
response = request.json()

print(
    json.dumps(
        response, sort_keys=True, indent=4, ensure_ascii=False, separators=(",", ": ")
    )
)
