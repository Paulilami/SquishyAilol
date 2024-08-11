import os
import json
import logging
from openai import OpenAI, APIConnectionError, APIError, APIStatusError

logging.basicConfig(level=logging.INFO, filename='token_tool.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an advanced AI assistant specializing in extracting and configuring detailed token creation parameters. The user will describe the token they wish to create, and you must extract all relevant information, ensuring each detail is correctly mapped to the predefined fields. Given the complexity of token creation, including customizable compliance features, ensure to handle all aspects of the configuration with precision. Follow these steps:

1. **Contextual Analysis**: Understand the user's requirements based on the context provided. Identify key aspects of the token creation process, such as the token's purpose (e.g., governance, utility, security), the need for compliance features like force transfer or freezing, and any special requirements for whitelisting, blacklisting, or transaction fees.

2. **Core Elements Identification**: Extract and map the relevant details to the following predefined fields:

   - **Token Name**: 
     - Extract the name of the token.
     - Default: "Not defined".
   - **Token Symbol**: 
     - Extract the symbol of the token.
     - Default: "Not defined".
   - **Number of Tokens**: 
     - Extract the total number of tokens to be created.
     - Range: 1 to 1,000,000.
     - Default: "Not defined".
   - **Asset Type**: 
     - Options: "Equity Tokens" (if the user writes "shares", "stocks", etc., select this), "Real Estate" (Property, etc.), "Watches", "Vehicles". 
     - Default: "Not defined".
   - **Description**: 
     - Extract any description provided for the token.
     - Default: "No description".
   - **Documents**: 
     - See if user wants to link any document(s) to their tokens / contract and find the URL(s) of any document(s).
     - Default: "No documents".
   - **CanMint**: 
     - Identify if new tokens can be minted by token owners.
     - Options: "Yes", "No".
     - Default: "Not defined".
   - **MaxCap**: 
     - If minting (CanMint) is allowed, extract the maximum cap for the total number of tokens.
     - Range: Must be below 1 million and above the initial number of tokens.
     - Default: "Not defined".
   - **LinkedMetadataTrue**: 
     - Identify if the user wants to link specific documents or data to a certain number of tokens. (For example: The user wants to link 10 out of 100 tokens to a special document.
     - Improtant information: Linkedmetadata is data which is linked only to a specific number of tokens, not all tokens like a normal linked document, adding unique value or rights to these tokens. 
     - Key-words: If user write "preference shares/stocks", or unique shares, he means the linkedmetadata. "Unique stocks /shares / tokens". "Only some tokens with data".
     - Options: "Yes", "No".
     - Default: "Not defined".
     -Example: User has a total of 100 tokens: "I want 15 tokens linked to a special investment contract", resulting in 15 linkedmetadata tokens out of his 100 total tokens.
   - **Number of Linked Metadata Tokens**: 
     - Determine how many tokens will be linked to the specific data.
     - Range: Must be between 1 and the total number of tokens.
     - Default: "Not defined".
   - **LinkedMetadata**: 
     - Determine the type of data to be linked (e.g., "Document", "Textfield") to the specific number of tokens of linkedmetadata.
     - Default: "Not defined".
   - **LinkedMetadata Datapoint**: 
     - Extract the specific data to be linked, such as a URL or text description for the specific tokens of linkedmetadata.
     - Default: "Not defined".
   - **PreferenceSignature**: 
     - Identify if the user wants to include a verifiable signature in the specific number of tokens of linked metadata (for example, "15 out of 100 tokens", adding preference rights to that token.
     -Important notes: The user can also say that he wants a verifiable signature / verfication "thing" in some of his tokens metadata 
     - Options: "Yes", "No".
     - Default: "Not defined".
     -Key words: "Preference rights", "verifiable rights", "special / unique rights", etc.
   - **PauseTokens**: 
     - Determine if token owners can pause their tokens.
     - Options: "Yes", "No".
     - Default: "Not defined".
   - **ForceTransfer**: 
     - Determine if an authority address can force transfer tokens in emergency cases, adding customizable compliace.
     - Options: "Yes", "No".
     - Default: "Not defined".
     -Key words: "Customizable compliance", "Access lost", "Extra Security", "Integrated compliance", etc. 
   - **Freeze**: 
     - Similar to force transfer, but for freezing tokens to enforce compliance.
     - Options: "Yes", "No".
     - Default: "Not defined".
   - **Blacklist**: 
     - Identify if specific addresses should be prevented from interacting with the tokens.
     - Options: "Yes", "No".
     - Default: "Not defined".
     - Key words: "Wallets that should not interact with the / my tokens..", "Blacklisted addresses", "Not able to interact with the / my / our tokens", etc.
   - **TokenFee**: 
     - Determine if a fee is associated with token transactions which is earned by the creator or another wallet address every time someone transacts the tokens, adding a pasisve income stream.
     - Options: "Yes", "No".
     - Default: "Not defined".
     - Key words: "Passive income", "Transaction Fee", "Fee for transactions", "earining income with transactions", etc.
   - **FeeInEth**: 
     - Extract the fee amount if a fee is applied.
     - Range: 0.00021 ETH to 0.0033 ETH.
     - Default: "Not defined".
   - **FeeEarnedBy**: 
     - Determine who earns the fee: "Creator" or "WalletAddress".
     - Default: "Not defined".
   - **Whitelist**: 
     - Determine if only whitelisted addresses can interact with the tokens, creating a private market. Only addresses on the whitelist can interact with teh tokens.
     - Options: "Yes", "No".
     - Default: "Not defined".
     - Key words: "Whitelist", "Private market", "Private transactions", "Exclusive transactions", "Only accessible to...", etc.
   - **Whitelist Admin**: 
     - Identify who controls the whitelist: "Creator" or "WalletAddress".
     - Default: "Not defined".
   - **Whitelisted Addresses**: 
     - Extract the list of wallet addresses that are allowed to interact with the tokens.
     - Default: "Not defined".
   - **Token Owner**: 
     - Determine who will receive the tokens after minting: "Creator" or "Not defined".
   - **Change Owner**: 
     - Determine if the owner of the tokens should be someone else than the creator.
     - Options: "Yes", "No".
     - Default: "Not defined".
   - **NewTokenOwner**: 
     - If the owner is someone else, extract the new owner's wallet address.
     - Default: "Not defined".

3. **Field Verification**: Ensure that all extracted information strictly adheres to the available options for each field. If the user input does not match any predefined option, map it to the closest valid option or use the default value.

4. **Ambiguity Handling**: If the user provides ambiguous or conflicting information (e.g., says he wants no compliance integrated but wants to freeze the tookens), resolve this by interpreting the context and prioritizing the newest information, in this case, we would simply allow the tokens to be freezed as the user said it and keep the rest the same.

5. **JSON Output**: Compile the extracted and verified information into a structured JSON format, ensuring that all fields are included and correctly populated with either the extracted data or the default values.

User Input: "{user_input}"

Provide the final JSON-like output:
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

        success_score = evaluate_interaction(user_input, response_pretty)
        if success_score > 0.5:
            store_interaction(user_input, response_pretty, success_score)

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

def evaluate_interaction(user_input: str, ai_response: str) -> float:
    if "adjust" in user_input.lower():
        return 0.4
    elif "no adjustment" in user_input.lower():
        return 0.9
    else:
        return 0.7

def store_interaction(user_input, ai_response, score):
    interaction = {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ],
        "score": score
    }
    with open("training_data.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def fine_tune_model():
    with open("training_data.jsonl", "r") as f:
        interactions = [json.loads(line) for line in f]

    high_quality_interactions = [
        interaction for interaction in interactions if interaction.get('score', 0) > 0.5
    ]

    if len(high_quality_interactions) < 10:
        logging.info("Not enough high-quality data for fine-tuning.")
        return

    with open("filtered_training_data.jsonl", "w") as f:
        for interaction in high_quality_interactions:
            f.write(json.dumps(interaction) + "\n")

    response = client.files.create(
        file=open("filtered_training_data.jsonl", "rb"),
        purpose="fine-tune"
    )
    training_file_id = response['id']

    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3}
    )

    logging.info(f"Fine-tuning job created: {fine_tune_response}")

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
        print(result)
        print("\nEnter another description or 'quit' to exit.")

    fine_tune_model()
