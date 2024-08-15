import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

logging.basicConfig(level=logging.INFO, filename='token_tool.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROTOCOL_FIELDS = {
    "Token Name": ["Not defined"],
    "Token Symbol": ["Not defined"],
    "Number of Tokens": ["Not defined"],
    "Asset Type": ["Equity Tokens", "Debt Tokens", "Real Estate", "Pokemon Cards", "Commodities", "Watches", "Vehicles", "Not defined"],
    "Description": ["No description"],
    "UnifiedData": ["True", "False"],
    "UnifiedDataIndex": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "UnifiedDataType": ["Document", "Text Field", "Link", "Not provided"],
    "UnifiedDataName": ["Not defined"],
    "UnifiedDataPoint": ["Not defined"],
    "CanMint": ["True", "False"],
    "MaxCap": ["Not defined"],
    "LinkedData": ["True", "False"],
    "LinkedDataIndex": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "LinkedDataType": ["Document", "Text Field", "Link", "Not provided"],
    "NumberofLinkedDataTokens": "Not defined",
    "LinkedDataName": ["Not defined"],
    "LinkedDataPoint": ["Not defined"],
    "PreferenceSignature": ["True", "False"],
    "PauseTokens": ["True", "False"],
    "ForceTransfer": ["True", "False"],
    "Freeze": ["True", "False"],
    "Blacklist": ["True", "False"],
    "TokenFee": ["True", "False"],
    "FeeEarnedBy": ["Creator", "Wallet Address", "Not defined"],
    "Whitelist": ["True", "False"],
    "Whitelist Admin": ["Creator", "Wallet Address", "Not defined"],
    "TokenOwner": ["Creator", "Wallet Address", "Not defined"]
}

def sanitize_output(config):
    """Ensure that the output configuration strictly adheres to the predefined format."""
    sanitized_config = {}
    for key in PROTOCOL_FIELDS:
        if key in config:
            if isinstance(PROTOCOL_FIELDS[key][0], list): 
                sanitized_config[key] = config.get(key, [])
            else:
                sanitized_config[key] = config.get(key, PROTOCOL_FIELDS[key][0])
        else:
            sanitized_config[key] = PROTOCOL_FIELDS[key][0]
    return sanitized_config

def update_unified_data(user_input: str, current_config: dict):
    """Update Unified Data by adding a new document to the configuration."""
    unified_data_indices = current_config.get("UnifiedDataIndex", [])
    unified_data_types = current_config.get("UnifiedDataType", [])
    unified_data_names = current_config.get("UnifiedDataName", [])
    unified_data_points = current_config.get("UnifiedDataPoint", [])

    if not isinstance(unified_data_indices, list):
        unified_data_indices = [unified_data_indices]
    if not isinstance(unified_data_types, list):
        unified_data_types = [unified_data_types]
    if not isinstance(unified_data_names, list):
        unified_data_names = [unified_data_names]
    if not isinstance(unified_data_points, list):
        unified_data_points = [unified_data_points]

    current_index = len(unified_data_indices) + 1 

    doc_name, doc_url = extract_document_info(user_input)

    current_config["UnifiedData"] = "True"
    current_config["UnifiedDataIndex"] = unified_data_indices + [str(current_index)]
    current_config["UnifiedDataType"] = unified_data_types + ["Document"]
    current_config["UnifiedDataName"] = unified_data_names + [doc_name]
    current_config["UnifiedDataPoint"] = unified_data_points + [doc_url]

    return current_config

def extract_document_info(user_input: str):
    """Extract document name and URL from user input."""
    if "room plans" in user_input.lower():
        return "Room Plans", "https://www.example.urls/to/insert"
    elif "investment contract" in user_input.lower():
        return "Investment Contract", "https://www.example.urls/to/insert/two"
    elif "legal rights" in user_input.lower():
        return "Legal Rights", "https://example.com/rights"
    else:
        return "Undefined Document", "https://undefined.url"

def create_or_update_token_config(user_input: str, current_config: dict) -> dict:
    try:
        if "linked to" in user_input.lower() and "all tokens" not in user_input.lower():
            current_config["LinkedData"] = "True"
        else:
            current_config = update_unified_data(user_input, current_config)

        if "integrated compliance" in user_input.lower():
            current_config["PauseTokens"] = "True"
            current_config["ForceTransfer"] = "True"
            current_config["Freeze"] = "True"
            current_config["Blacklist"] = "True"

        prompt = f"""
Given the current configuration:
{json.dumps(current_config, indent=2)}

And the user input:
"{user_input}"

Update the configuration based on the user input. Provide ONLY the updated JSON configuration as your response, with no additional text:
"""

        client = OpenAI()

        messages = [
            {"role": "system", "content": "You are a specialized AI assistant for updating token configurations."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        response_text = response.choices[0].message.content.strip()

        try:
            updated_config = json.loads(response_text)
            updated_config = sanitize_output(updated_config)
            return updated_config
        except json.JSONDecodeError:
            print("Failed to update configuration. Keeping current configuration.")
            return current_config

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return current_config

def evaluate_interaction(user_input: str, config: dict) -> float:
    """Evaluate the quality of the interaction."""
    if "adjust" in user_input.lower():
        return 0.4
    elif "no adjustment" in user_input.lower():
        return 0.9
    else:
        return 0.7

def store_interaction(user_input, config, score):
    """Store the interaction for future fine-tuning."""
    interaction = {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(config)}
        ],
        "score": score
    }
    with open("training_data_token_tool.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def fine_tune_model():
    """Fine-tune the model using stored interactions."""
    with open("training_data_token_tool.jsonl", "r") as f:
        interactions = [json.loads(line) for line in f]

    high_quality_interactions = [
        interaction for interaction in interactions if interaction.get('score', 0) > 0.5
    ]

    if len(high_quality_interactions) < 10:
        logging.info("Not enough high-quality data for fine-tuning.")
        return

    with open("filtered_training_data_token_tool.jsonl", "w") as f:
        for interaction in high_quality_interactions:
            f.write(json.dumps(interaction) + "\n")

    response = client.files.create(
        file=open("filtered_training_data_token_tool.jsonl", "rb"),
        purpose="fine-tune"
    )
    training_file_id = response.id

    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini",
        hyperparameters={"n_epochs": 3}
    )

    logging.info(f"Fine-tuning job created: {fine_tune_response}")

def handle_user_input(current_config):
    """Handle user input and update the configuration accordingly."""
    while True:
        user_input = input("Enter changes you want to make (or 'done' to finish and 'quit' to stop): ")

        if user_input.lower() == 'done':
            break
        elif user_input.lower() == 'quit':
            return None, False

        updated_config = create_or_update_token_config(user_input, current_config)
        print(json.dumps(updated_config, indent=2))
        current_config = updated_config

    return current_config, True

if __name__ == '__main__':
    current_config = {
        "Token Name": "Not defined",
        "Token Symbol": "Not defined",
        "Number of Tokens": "Not defined",
        "Asset Type": "Not defined",
        "Description": "No description",
        "UnifiedData": "False",
        "UnifiedDataIndex": [],
        "UnifiedDataType": [],
        "UnifiedDataName": [],
        "UnifiedDataPoint": [],
        "CanMint": "False",
        "MaxCap": "Not defined",
        "LinkedData": "False",
        "LinkedDataIndex": "1",
        "LinkedDataType": "Not provided",
        "NumberofLinkedDataTokens": "Not defined",
        "LinkedDataName": "Not defined",
        "LinkedDataPoint": "Not defined",
        "PreferenceSignature": "False",
        "PauseTokens": "False",
        "ForceTransfer": "False",
        "Freeze": "False",
        "Blacklist": "False",
        "TokenFee": "False",
        "FeeEarnedBy": "Not defined",
        "Whitelist": "False",
        "Whitelist Admin": "Not defined",
        "TokenOwner": "Creator"
    }

    final_config, _ = handle_user_input(current_config)

    print("\nFinal Configuration:")
    print(json.dumps(final_config, indent=2))
