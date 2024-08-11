# OpenAI Payment Streams, Token Vaults, and Token Tool Configuration

This repository provides Python scripts utilizing OpenAI's API for processing payment streams, token vaults, and token creation configurations. Designed for developers in finance and token management, these scripts automate complex configurations and output structured JSON.

## Features

- **OpenAI API Integration**: Leverages NLP capabilities for processing complex inputs.
- **JSON Input/Output**: Handles user inputs and outputs structured JSON data.
- **Automatic Fine-Tuning**: Enhances model accuracy based on user interactions.
- **Language**: Python

## Getting Started

### Prerequisites

- Python 3.x
- OpenAI API key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repository.git
    cd your-repository
    ```

2. Install the required Python packages:
    ```bash
    pip install openai
    ```

3. Set your OpenAI API key:
    ```bash
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

## Usage

### Payment Streams

#### Script: `ai_payments.py`

Execute the script:
```bash
python3 ai_payments.py
```

## Testing Prompts

1. *Simple*:
    ```bash
    Create a monthly payment stream with fixed $1000 payments.
    ```

2. *Intermediate*:
    ```bash
    Generate a payment stream for my company’s preferred shares with semi-annual payouts linked to performance metrics.
    ```

3. *Complex*:
    ```bash
    Create a dividend payment stream for my equity tokens with a quarterly payout of the sum that got paid into the payment stream contract.
    ```

    ### Payment Streams

#### Script: `ai_payments.py`

Execute the script:
```bash
python3 ai_payments.py
```

## Testing Prompts

1. *Simple*:
    ```bash
    Set up a 12-month token vault for my savings.
    ```

2. *Intermediate*:
    ```bash
    Create a token vault for my cryptocurrency holdings with a 2-year lock-up and a 10% withdrawal penalty before maturity.
    ```

3. *Complex*:
    ```bash
    Configure a token vault for my real estate assets with a 24-month duration and a 5% early withdrawal penalty.
    ```

    ### Payment Streams

#### Script: `ai_payments.py`

Execute the script:
```bash
python3 ai_payments.py
```
## Testing Prompts

1. *Simple*:
    ```bash
    Create 1000 tokens named "MyToken" with symbol "MTK".
    ```

2. *Intermediate*:
    ```bash
    Create 10,000 tokens called "SecureToken" with symbol "STK", each linked to a security document.
    ```

3. *Complex*:
    ```python
    I want to create 500,000 tokens called 'EquityToken' with the symbol 'EQT'. The tokens should be linked to a shareholder agreement document, and 10,000 of these tokens should be linked to a signed contract document. New tokens can be minted up to a max cap of 600,000. These tokens should have the ability to freeze in case of regulatory issues, and an authority address must be able to force transfer tokens in case of fraud. I want to allow a 0.0015 ETH transaction fee, which I will earn. Only whitelisted addresses should be able to interact with the tokens, and I'll manage the whitelist as the admin. The primary token owner will be my wallet address. Also, the tokens need to have a preference signature to create preference shares. Lastly, ensure that token owners can pause their tokens if needed.
    ```

# Output

Upon running the scripts, you'll be prompted to enter descriptions for your payment stream, token vault, or token configuration. The script will then generate and return a structured JSON output.

## Example Output

```json
{
    "Payer": "Creator",
    "Input Payment Frequency": "Quarterly",
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
```

# Fine-Tuning

The scripts include automatic fine-tuning features, continuously improving AI accuracy by learning from user interactions.

