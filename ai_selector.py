import os
import json
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, filename='protocol.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an advanced AI assistant specializing in protocol classification for financial and digital asset management tasks. Your role is to accurately determine the most appropriate protocol based on user input. Analyze the input carefully and select the best-matching protocol from the options below.

Available Protocols:
1. ai_payments.py
2. ai_vaults.py
3. ai_tokentool.py

Protocol Descriptions:
• ai_payments.py:
  - Purpose: Manages payment streams, recurring payments, and financial transactions
  - Key concepts: Payment frequency, amounts, distributions, payers, currencies (ETH, EUR)
  - Typical keywords: payment stream, dividend, payout, recurring payment, ETH, EUR

• ai_vaults.py:
  - Purpose: Handles digital token vaults and secure asset storage
  - Key concepts: Asset storage, access control, penalties, lockup periods
  - Typical keywords: token vault, secure space, digital safe, penalty, lockup period

• ai_tokentool.py:
  - Purpose: Manages token creation, minting, and compliance features
  - Key concepts: Token creation, metadata, minting, compliance (freezing, force transfer)
  - Typical keywords: token creation, mint tokens, metadata, freeze tokens, whitelist

Classification Guidelines:
1. Analyze the overall context of the user's request.
2. Identify key concepts and keywords that align with a specific protocol.
3. Consider overlapping features (e.g., payment streams in vaults) and prioritize based on the primary purpose.
4. Select the most appropriate protocol based on your analysis.

User Input: {user_input}

Respond ONLY with a JSON object in the following format:
{{
  "Target": "<protocol_name>",
  "Prompt": "{user_input}"
}}

Where <protocol_name> is one of: ai_payments.py, ai_vaults.py, or ai_tokentool.py

Do not include any explanations, additional text, or formatting outside of this JSON structure.
"""

def determine_protocol(user_input: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a protocol classifier."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(user_input=user_input)}
            ]
        )
        
        api_response = json.loads(response.choices[0].message.content.strip())
        
        if "Target" not in api_response or "Prompt" not in api_response:
            raise ValueError("Invalid response structure from API")
        
        valid_protocols = ["ai_payments.py", "ai_vaults.py", "ai_tokentool.py"]
        if api_response["Target"] not in valid_protocols:
            raise ValueError(f"Invalid protocol: {api_response['Target']}")
        
        logging.info(f"User Input: {user_input}")
        logging.info(f"Classified Protocol: {api_response['Target']}")
        
        return json.dumps(api_response, indent=2)
    
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {str(e)}")
        return json.dumps({"error": "Invalid JSON response from API"})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return json.dumps({"error": str(e)})

def handle_user_input():
    user_input = input("Describe your task: ")
    return determine_protocol(user_input)

if __name__ == '__main__':
    result = handle_user_input()
    print(result)
