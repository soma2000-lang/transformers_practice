from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import os
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
import time
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def generate_security_logs_batch(num_logs=10):
    """Generate security logs using GPT-4."""
    
    prompt = f"""Generate {num_logs} different security log entries. Include a mix of:
    1. Windows Event logs
    2. Linux syslog entries
    3. Firewall logs
    4. Authentication logs
    5. Network access logs
    6. Web server logs
    7. DNS query logs
    8. Database access logs
    
    Make them look realistic with:
    - Real IP addresses
    - Timestamps
    - Process IDs
    - User IDs
    - File paths
    - Commands
    - Error codes
    - Session IDs
    
    Return ONLY the log entries, one per line, nothing else.
    Make them diverse and realistic."""

    response = completion_with_backoff(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a security log generator. Generate realistic security logs that could come from various systems and services."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip().split('\n')

def create_training_data(num_batches=50, logs_per_batch=10, output_file="security_logs.txt"):
    """Create training data file with GPT-4 generated security logs."""
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Found existing training data at {output_file}")
        # Count number of lines in existing file
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"File contains {line_count} log entries")
        return output_file
    
    total_logs = []
    print(f"Generating {num_batches * logs_per_batch} security logs...")
    
    for _ in tqdm(range(num_batches)):
        logs = generate_security_logs_batch(logs_per_batch)
        total_logs.extend(logs)
        # Sleep to respect rate limits
        time.sleep(1)
    
    # Write logs to file
    with open(output_file, "w", encoding="utf-8") as f:
        for log in total_logs:
            f.write(log.strip() + "\n")
    
    print(f"Generated {len(total_logs)} logs and saved to {output_file}")
    return output_file

def train_tokenizer(training_file, vocab_size=30000, save_dir="security_tokenizer"):
    """Train a BPE tokenizer on security logs."""
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Configure the training
    print(f"\nTraining tokenizer on {training_file}...")
    tokenizer.train(
        files=[training_file],
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )
    
    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the tokenizer
    tokenizer.save_model(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    
    return tokenizer

def test_tokenizer(tokenizer, text):
    """Test the tokenizer on a sample text."""
    encoded = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Encoded tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    return encoded

def main():
    # Generate or load training data
    training_file = create_training_data(num_batches=50, logs_per_batch=10)
    
    # Train tokenizer
    tokenizer = train_tokenizer(training_file)
    print("Tokenizer training completed")
    
    # Test the tokenizer with different types of logs
    test_logs = [
        'May 15 10:23:45 server sshd[12345]: Failed password for invalid user admin from 192.168.1.100 port 43215 ssh2',
        'Microsoft-Windows-Security-Auditing: User Account created: Account Name: JohnDoe Status: Success',
        '192.168.1.50 - - [15/May/2024:10:24:33 +0000] "GET /admin/login.php HTTP/1.1" 401 287'
    ]
    
    print("\nTesting tokenizer on sample logs:")
    for log in test_logs:
        test_tokenizer(tokenizer, log)
    
    print("\nTokenizer training and testing completed!")

if __name__ == "__main__":
    main()
