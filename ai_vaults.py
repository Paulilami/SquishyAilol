import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

logging.basicConfig(level=logging.INFO, filename='token_vaults.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an AI assistant that extracts relevant token vault configuration data from user input. The user will describe a token vault, and you need to extract the following information:

1. Asset Type: Real estate, Watches, Equity shares, Vehicles. Default: "Not defined".
2. Access Control: All token owners, Only admin (admin is used when the user says "Myself" as the creator is the admin), Admin & Managers (if user said that managers have access control, the admin also has it as he is over the managers), Whitelisted addresses (user chooses specific addresses). Default: "Not defined".
3. Duration: 1 month minimum up to 36 months. Default: "Not defined".
4. Penalty: between 0.01% and 10%. Default: "Not defined".
5. Input Payments: Yes, No. Default: "Not defined".
6. Input Payments Frequency: Daily, Weekly, Monthly, Quarterly, Half-yearly, Yearly. Default: "Not defined".
7. Input Payment Currency: Not defined, EUR, USD, ETH, Other. Default: "Not defined".
8. Output Payment Distribution: Yes, No. Default: "Not defined".
9. Distribution Frequency: Daily, Weekly, Monthly, Quarterly, Half-yearly, Yearly. Default: "Not defined".
10. Distribute to: All token holders/owners, only whitelisted addresses. Default: "Not defined".
11. Vault Description: No description, <description>. Default: "No description".
12. Admin: Always set to "Creator".
13. Managers: Whitelisted addresses or "Not defined".
14. Manager Permissions: Input payments, change data, withdraw funds, delete payment stream. Default: "Not defined".

Given this structure, extract the relevant information from the following input and provide the output:

"{user_input}"

Output the extracted data in a JSON-like structure. If any data points are not specified, return the default value.
"""

def create_token_vault(user_input: str) -> str:
    try:
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
            response_pretty = json.dumps(response_json, indent=2)
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
    user_input = input("Enter your token vault description (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        return None

    return create_token_vault(user_input)

if __name__ == '__main__':
    while True:
        result = handle_user_input()
        if result is None:
            break

        print("\nProcessed Token Vault Configuration:")
        print(result)
        print("\nEnter another description or 'quit' to exit.")
