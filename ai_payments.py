import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

logging.basicConfig(level=logging.INFO, filename='payment_streams.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an AI assistant that specializes in extracting and processing complex payment stream configurations. The user will describe a payment stream, and you will perform the following tasks step by step:

1. Identify and list the core elements of the payment stream.
2. Extract the following information:
   -    - Payer: Creator (if the user writes "myself", select this), Everyone, Other address. Default: "Creator".
   - Input Payment Frequency: Daily, Weekly, Monthly, Quarterly, Half-yearly, Yearly. Default: "Not defined".
   - Input Payment Amount: Amount in ETH, Amount in EUR, or "Not defined".
   - Output Payment Distribution: Yes, No. Default: "Not defined".
   - Distribution Frequency: Daily, Weekly, Monthly, Quarterly, Half-yearly, Yearly. Default: "Not defined".
   - Distribute to: All token holders/owners, only whitelisted addresses. Default: "Not defined".
   - Pause Payments: Yes, No. Default: "Not defined".
   - Pause Payments by: Creator/Myself, Whitelisted addresses. Default: "Not defined".
   - Admin: Always set to "Creator".
   - Managers: Whitelisted addresses or "Not defined".
   - Manager Permissions: Input payments, change data, withdraw funds, delete payment stream. Default: "Not defined".

3. Compile this information into a structured JSON format.
4. If any data points are not specified, return the default value.

User Input: "{user_input}"

Provide the final JSON-like output:
"""

def create_payment_stream(user_input: str) -> str:
    try:
        # Use advanced prompt to guide the model without fine-tuning
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(user_input=user_input)}
            ]
        )

        response_text = response.choices[0].message.content
        
        try:
            response_json = json.loads(response_text)
            response_pretty = json.dumps(response_json, indent=2)  # Pretty-print the JSON
        except json.JSONDecodeError:
            response_pretty = response_text

        logging.info(f"User Input: {user_input}")
        logging.info(f"AI Response: {response_pretty}")

        return response_pretty

    except APIConnectionError as e:
        logging.error(f"API Connection Error: {e.__cause__}")
        return '{"error": "API connection failed"}'
    except APIStatusError as e:
        logging.error(f"API Status Error: {e.status_code}, Response: {e.response}")
        return '{"error": "API status error occurred"}'
    except APIError as e:
        logging.error(f"General API Error: {e}")
        return '{"error": "API request failed"}'
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        return '{"error": "An unexpected error occurred"}'

def handle_user_input():
    user_input = input("Enter your payment stream description (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        return None

    return create_payment_stream(user_input)

if __name__ == '__main__':
    while True:
        result = handle_user_input()
        if result is None:
            break

        print("\nProcessed Payment Stream Configuration:")
        print(result)
        print("\nEnter another description or 'quit' to exit.")
