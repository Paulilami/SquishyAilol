import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging
import re

logging.basicConfig(level=logging.INFO, filename='token_vaults.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROTOCOL_FIELDS = {
    "Asset Type": ["Equity Tokens", "Real Estate", "Watches", "Vehicles", "Not defined"],
    "Access Control": ["All token owners", "Only admin", "Admin & Managers", "Whitelisted addresses", "Not defined"],
    "Duration": ["Not defined"],  
    "Penalty": ["Not defined"],  
    "Input Payments": ["Yes", "No", "Not defined"],
    "Input Payments Frequency": ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly", "Not defined"],
    "Input Payment Currency": ["EUR", "USD", "ETH", "Other", "Not defined"],
    "Output Payment Distribution": ["Yes", "No", "Not defined"],
    "Distribution Frequency": ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly", "Not defined"],
    "Distribute to": ["All token holders/owners", "Only whitelisted addresses", "Not defined"],
    "Vault Description": ["No description"],
    "Admin": ["Creator"],
    "Managers": ["Whitelisted addresses", "Creator", "Not defined"],
    "Manager Permissions": ["Input payments", "Change data", "Withdraw funds", "Delete payment stream", "Not defined"]
}

PROMPT_TEMPLATE = """
You are an advanced AI assistant specialized in configuring digital token vaults. Your task is to extract, analyze, and process detailed vault configurations from the user's input. The user may describe various scenarios related to asset storage, duration, penalties, payment streams, and access control. You must interpret the context and determine the appropriate parameters for the vault configuration. When the user writes "my", "me", "I", "myself", etc, he means the (=) "Creator".

Ensure that all data extracted corresponds to the predefined fields and options:

{protocol_fields}

User's new input or requested changes:
{user_input}

Update the vault configuration based on the user's input. If a field is not mentioned or changed, keep its previous value from the current configuration. Provide ONLY ONE updated JSON-like output, strictly adhering to the predefined fields and options. Only include fields that have changed; for unchanged fields, do not include them in the output. Do not include any explanations or additional text, just the single, final JSON object with the changes:
"""

def validate_config(config):
    """Validate and correct the configuration based on protocol fields."""
    for field, value in config.items():
        if field == "Duration":
            if value != "Not defined":
                try:
                    months = int(value.split()[0])
                    if months < 1 or months > 36:
                        config[field] = "Not defined"
                except:
                    config[field] = "Not defined"
        elif field == "Penalty":
            if value != "Not defined":
                try:
                    penalty = float(value[:-1])
                    if penalty < 0.01 or penalty > 10:
                        config[field] = "Not defined"
                except:
                    config[field] = "Not defined"
        elif field in PROTOCOL_FIELDS:
            if value not in PROTOCOL_FIELDS[field]:
                config[field] = "Not defined"
    return config

def create_or_update_token_vault(user_input: str, current_config: dict) -> dict:
    """Create or update the token vault configuration based on user input."""
    try:
        messages = [
            {"role": "system", "content": "You are a specialized AI assistant."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                protocol_fields=json.dumps(PROTOCOL_FIELDS, indent=2),
                current_config=json.dumps(current_config, indent=2),
                user_input=user_input
            )}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        response_text = response.choices[0].message.content.strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                updated_config = json.loads(json_str)
                for key, value in updated_config.items():
                    if value != "Not defined":
                        current_config[key] = value
                validated_config = validate_config(current_config)
                logging.info(f"User Input: {user_input}")
                logging.info(f"Updated Config: {json.dumps(validated_config, indent=2)}")
                return validated_config
            except json.JSONDecodeError:
                logging.error(f"Failed to parse extracted JSON: {json_str}")
        else:
            logging.error(f"No valid JSON found in the response: {response_text}")
        
        return current_config

    except (APIConnectionError, APIStatusError, APIError) as e:
        logging.error(f"API Error: {str(e)}")
        return current_config
    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
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
    with open("training_data_vaults.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def fine_tune_model():
    """Fine-tune the model using stored interactions."""
    with open("training_data_vaults.jsonl", "r") as f:
        interactions = [json.loads(line) for line in f]

    high_quality_interactions = [
        interaction for interaction in interactions if interaction.get('score', 0) > 0.5
    ]

    if len(high_quality_interactions) < 10:
        logging.info("Not enough high-quality data for fine-tuning.")
        return

    with open("filtered_training_data_vaults.jsonl", "w") as f:
        for interaction in high_quality_interactions:
            f.write(json.dumps(interaction) + "\n")

    response = client.files.create(
        file=open("filtered_training_data_vaults.jsonl", "rb"),
        purpose="fine-tune"
    )
    training_file_id = response.id

    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3}
    )

    logging.info(f"Fine-tuning job created: {fine_tune_response}")

def handle_user_input(current_config, is_first_input):
    """Handle user input and update the configuration accordingly."""
    if is_first_input:
        user_input = input("Enter your initial token vault description: ")
    else:
        user_input = input("Enter changes you want to make or 'done' to finish and 'quit' to stop: ")

    if user_input.lower() == 'done':
        return None, True
    elif user_input.lower() == 'quit':
        return None, False

    updated_config = create_or_update_token_vault(user_input, current_config)
    success_score = evaluate_interaction(user_input, updated_config)
    
    if success_score > 0.5:
        store_interaction(user_input, updated_config, success_score)

    return updated_config, None

if __name__ == '__main__':
    current_config = {
        "Asset Type": "Not defined",
        "Access Control": "Not defined",
        "Duration": "Not defined",
        "Penalty": "Not defined",
        "Input Payments": "Not defined",
        "Input Payments Frequency": "Not defined",
        "Input Payment Currency": "Not defined",
        "Output Payment Distribution": "Not defined",
        "Distribution Frequency": "Not defined",
        "Distribute to": "Not defined",
        "Vault Description": "No description",
        "Admin": "Creator",
        "Managers": "Not defined",
        "Manager Permissions": "Not defined"
    }
    is_successful = None
    is_first_input = True

    while is_successful is None:
        result, is_successful = handle_user_input(current_config, is_first_input)
        if result is not None:
            current_config = result
            print(json.dumps(current_config, indent=2))
        is_first_input = False

    if is_successful:
        print("Generation was successful. Proceeding with fine-tuning.")
        fine_tune_model()
    else:
        print("Process terminated. No data will be used for training or fine-tuning.")
