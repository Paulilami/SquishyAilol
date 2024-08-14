import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='payment_streams.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROTOCOL_FIELDS = {
    "Asset Type": ["Equity Tokens", "Real Estate", "Watches", "Vehicles", "Not defined"],
    "Payer": ["Creator", "Everyone", "Other address"],
    "Input Payment Frequency": ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly", "Not defined"],
    "Input Payment Amount": ["Not defined"],  # handled separately
    "Output Payment Distribution": ["Yes", "No", "Not defined"],
    "Distribution Frequency": ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly", "Not defined"],
    "Distribute to": ["All token holders/owners", "Only whitelisted addresses", "Not defined"],
    "Pause Payments": ["Yes", "No", "Not defined"],
    "Pause Payments by": ["Creator/Myself", "Whitelisted addresses", "Not defined"],
    "Admin": ["Creator"],
    "Managers": ["Whitelisted addresses", "Creator" "Not defined"],
    "Manager Permissions": ["Input payments", "Change data", "Withdraw funds", "Delete payment stream", "Not defined"]
}

PROMPT_TEMPLATE = """
You are an advanced AI assistant specialized in financial payment streams. Your task is to extract, analyze, and process detailed payment stream configurations from the user's input. The user may describe a variety of financial use cases, including but not limited to dividend payments, subscriptions, rent, salaries, and other recurring or one-time payment structures. You must interpret the financial context and determine the appropriate parameters for the payment stream. When the user writes "my", "me", "I", "myself", etc, he means the (=) "Creator". Meaning user = "Creator", do not write "User".

Imprtant info: Please always think about the context of the prompt, for example, when the user writes he wants a dividend payment, he for sure has an output payment in a not determined frequency and an input payment which can be made at any time etc. think about what payment stream types result in what outcome and think logically to provide the right data output.

Ensure that all data extracted corresponds to the predefined fields and options:

{protocol_fields}

User's new input or requested changes:
{user_input}

Update the payment stream configuration based on the user's input. If a field is not mentioned or changed, keep its previous value. Provide ONLY ONE updated JSON-like output, strictly adhering to the predefined fields and options. Do not include any explanations or additional text, just the single, final JSON object:
"""

def validate_config(config):
    """Validate and correct the configuration based on protocol fields."""
    for field, value in config.items():
        if field == "Input Payment Amount":
            if value != "Not defined" and not (value.startswith("ETH ") or value.startswith("EUR ")):
                config[field] = "Not defined"
        elif field in PROTOCOL_FIELDS:
            if value not in PROTOCOL_FIELDS[field]:
                config[field] = "Not defined"
    return config

def create_or_update_payment_stream(user_input: str, current_config: dict) -> dict:
    try:
        client = OpenAI()
        
        prompt = f"""
Given the current configuration:
{json.dumps(current_config, indent=2)}

And the user input:
"{user_input}"

Update the configuration based on the user input. Provide ONLY the updated JSON configuration as your response, with no additional text:
"""

        messages = [
            {"role": "system", "content": "You are a specialized AI assistant for updating payment stream configurations."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        response_text = response.choices[0].message.content.strip()
        
        try:
            updated_config = json.loads(response_text)
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
    with open("training_data_streams.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def fine_tune_model():
    """Fine-tune the model using stored interactions."""
    with open("training_data_streams.jsonl", "r") as f:
        interactions = [json.loads(line) for line in f]

    high_quality_interactions = [
        interaction for interaction in interactions if interaction.get('score', 0) > 0.5
    ]

    if len(high_quality_interactions) < 10:
        logging.info("Not enough high-quality data for fine-tuning.")
        return

    with open("filtered_training_data_streams.jsonl", "w") as f:
        for interaction in high_quality_interactions:
            f.write(json.dumps(interaction) + "\n")

    response = client.files.create(
        file=open("filtered_training_data_streams.jsonl", "rb"),
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
        user_input = input("Enter your initial payment stream description: ")
    else:
        user_input = input("Enter changes you want to make or 'done' to finish and 'quit' to stop: ")

    if user_input.lower() == 'done':
        return None, True
    elif user_input.lower() == 'quit':
        return None, False

    updated_config = create_or_update_payment_stream(user_input, current_config)
    success_score = evaluate_interaction(user_input, updated_config)
    
    if success_score > 0.5:
        store_interaction(user_input, updated_config, success_score)

    return updated_config, None

if __name__ == '__main__':
    current_config = {
        "Asset Type": "Not defined",
        "Payer": "Creator",
        "Input Payment Frequency": "Not defined",
        "Input Payment Amount": "Not defined",
        "Output Payment Distribution": "Not defined",
        "Distribution Frequency": "Not defined",
        "Distribute to": "Not defined",
        "Pause Payments": "Not defined",
        "Pause Payments by": "Not defined",
        "Admin": "Creator",
        "Managers": "Not defined",
        "Manager Permissions": "Not defined"
    }

    while True:
        user_input = input("Enter changes you want to make (or 'done' to finish): ")
        
        if user_input.lower() == 'done':
            break
        
        updated_config = create_or_update_payment_stream(user_input, current_config)
        
        print(json.dumps(updated_config, indent=2))
        
        current_config = updated_config

    print("\nFinal Configuration:")
    print(json.dumps(current_config, indent=2))
