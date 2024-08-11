import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

logging.basicConfig(level=logging.INFO, filename='token_tool.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an AI assistant that extracts relevant token creation configuration data from user input. The user will describe the token they want to create, and you need to extract the following information:

1. Token Name: <NAME>, not defined.
2. Token Symbol: <SYMBOL>, not defined.
3. Number of Tokens: between 1-1000000. Default: "not defined".
4. Description: If empty, write "No description", if not empty, write the description from the user.
5. Documents: URL of document(s) linked to the contract. If empty, write "No documents".
6. CanMint: Yes or No. If new tokens can be minted by token owners.
7. MaxCap: If new tokens can be minted, we need a max cap for them. It must be below 1 million and above the number of tokens, otherwise "not defined".
8. LinkedMetadataTrue: If users want to link specific documents or specific data to a specific number of tokens, for example: link a signature document to 10 tokens. Yes or No.
9. Number of Linked Metadata Tokens: How many tokens will be linked to that specific data. Must be between 1 and the number of tokens, otherwise "not defined".
10. LinkedMetadata: The type of data that will be linked to the number of tokens: Document or Textfield.
11. LinkedMetadata Datapoint: The data they actually want to link. Either text like a description, or a URL for a document, otherwise "not defined".
12. PreferenceSignature: A signature allowing to add a verifiable reference signature, allowing the creation of verifiable preference shares, for example. Yes or No.
13. PauseTokens: If token owners can pause their tokens. Yes or No.
14. ForceTransfer: If an authority address can force transfer the tokens in an emergency case, for example, the user might write "I need compliance integrated when someone steals my tokens, then we need force transfer on yes". Yes or No.
15. Freeze: Exactly the same as force transfers, but for freezing tokens as more compliance integrated. Yes or No.
16. Blacklist: If users want to remove specific addresses from interacting with their tokens. Yes or No.
17. TokenFee: If users want to associate a fee to their tokens, anytime someone makes a transaction, the creator will earn this fee. Yes or No.
18. FeeInEth: Between 0.00021 ETH and 0.0033 ETH, otherwise "not defined".
19. FeeEarnedBy: Who will earn the fee: Creator or WalletAddress.
20. Whitelist: If the user wants to create a type of private market, where only whitelisted addresses can interact with their tokens. If "no", everyone can interact with their tokens. Yes or No.
21. Whitelist Admin: Creator or Wallet Address.
22. Whitelisted Addresses: Wallet addresses that can interact with the tokens.
23. Token Owner: The address that will receive the tokens after minting: Creator or not defined.
24. Change Owner: Yes or No.
25. NewTokenOwner: Wallet address.

Given this structure, extract the relevant information from the following input and provide the output:

"{user_input}"

Output the extracted data in a JSON-like structure. If any data points are not specified, return the default value.
"""

def create_token_tool_config(user_input: str) -> str:
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
    user_input = input("Enter your token creation configuration (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        return None

    return create_token_tool_config(user_input)

if __name__ == '__main__':
    while True:
        result = handle_user_input()
        if result is None:
            break

        print("\nProcessed Token Configuration:")
        print(result)
        print("\nEnter another description or 'quit' to exit.")
