import os
from openai import OpenAI, APIConnectionError, APIError, APIStatusError
import json
import logging

logging.basicConfig(level=logging.INFO, filename='payment_streams.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
You are an advanced AI assistant specialized in financial payment streams. Your task is to extract, analyze, and process detailed payment stream configurations from the user's input. The user may describe a variety of financial use cases, including but not limited to dividend payments, subscriptions, rent, salaries, and other recurring or one-time payment structures. You must interpret the financial context and determine the appropriate parameters for the payment stream. Ensure that all data extracted corresponds to the predefined fields. Follow these steps carefully:

1. **Contextual Analysis**: Understand the overall financial context, such as whether the payment is related to dividends, subscriptions, salaries, rent, or another financial mechanism. Identify whether the payment stream involves input payments (money flowing into the system) or output payments (distributions to recipients).

2. **Core Elements Identification and Update**: Identify and list the key elements of the payment stream, and ensure each is mapped to the following predefined fields. Update the existing configuration based on the new input. Identify the following fields:

    - **Asset Type**: 
     - Options: "Equity Tokens" (if the user writes "shares", "stocks", etc., select this), "Real Estate" (Property, etc.), "Watches", "Vehicles". 
     - Default: "Not defined".
   - **Payer**: 
     - Options: "Creator" (if the user writes "myself", select this), "Everyone", "Other address". 
     - Default: "Creator".
   - **Input Payment Frequency**: 
     - Options: "Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly". 
     - Default: "Not defined".
   - **Input Payment Amount**: 
     - Options: Amount in "ETH", Amount in "EUR", or "Not defined". 
     - Default: "Not defined".
   - **Output Payment Distribution**: 
     - Options: "Yes", "No". 
     - Default: "Not defined".
   - **Distribution Frequency**: 
     - Options: "Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly", "Yearly". 
     - Default: "Not defined".
   - **Distribute to**: 
     - Options: "All token holders/owners", "Only whitelisted addresses". 
     - Default: "Not defined".
   - **Pause Payments**: 
     - Options: "Yes", "No". 
     - Default: "Not defined".
   - **Pause Payments by**: 
     - Options: "Creator/Myself", "Whitelisted addresses". 
     - Default: "Not defined".
   - **Admin**: 
     - Always set to "Creator".
   - **Managers**: 
     - Options: "Whitelisted addresses" or "Not defined". 
     - Default: "Not defined".
   - **Manager Permissions**: 
     - Options: "Input payments", "Change data", "Withdraw funds", "Delete payment stream". 
     - Default: "Not defined".

3. **Field Verification**: Ensure that all extracted information strictly adheres to the available options for each field. If a user's input does not match any predefined option, map it to the closest valid option or use the default value.

4. **JSON Output**: Compile the extracted and verified information into a structured JSON format, ensuring that all fields are included and correctly populated with either the extracted data or the default values.

User Input: "{user_input}"

Existing Configuration (if any): {existing_data}

Provide the updated JSON-like output:
"""

def create_payment_stream(user_input: str) -> str:
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
    with open("training_data_streams.jsonl", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def fine_tune_model():
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
    training_file_id = response['id']

    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3}
    )

    logging.info(f"Fine-tuning job created: {fine_tune_response}")

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
        print(result)
        print("\nEnter another description or 'quit' to exit.")

    fine_tune_model()
