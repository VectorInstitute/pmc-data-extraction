"""
File: subcaption.py
----------------------
This script processes figure captions from a JSONL file, breaking them down into subcaptions
using a language model. It saves the results to a JSON file.
"""

import re
import json
import argparse
from sys import stderr
from tqdm import tqdm
from typing import List, Dict, Any

from openai import OpenAI


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each representing an item in the dataset.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f][20:23]


def process_caption(client: OpenAI, system_prompt: str, caption: str, model: str, max_tokens: int) -> str:
    """
    Process a caption using the language model.

    Args:
        client (OpenAI): OpenAI client instance.
        system_prompt (str): System prompt for the language model.
        caption (str): Caption to process.
        model (str): Model directory being used.
        max_tokens (int): Maximum number of tokens for the model response.

    Returns:
        str: Processed caption from the language model.
    """
    user_prompt = f"Caption: \n{caption}".strip()
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    
    return completion.choices[0].message.content


def parse_subcaptions(output: str) -> Dict[str, str]:
    """
    Parse the output from the language model into subcaptions.

    Args:
        output (str): Output from the language model.

    Returns:
        Dict[str, str]: Dictionary of subcaptions, where keys are subfigure labels and values are subcaptions.
    """
    lines = output.strip().split('\n')

    if not lines[0].upper().startswith("YES"):
        return {"Subfigure-A": '\n'.join(lines)}
    
    subcaptions = {}
    current_key = None
    current_value = []

    for line in lines[1:]:  # Skip the "YES" line
        match = re.match(r'^Subfigure-([A-Z]):\s*(.*)', line)

        if match:
            if current_key:
                subcaptions[current_key] = ' '.join(current_value).strip()
            current_key = f"Subfigure-{match.group(1)}"
            current_value = [match.group(2)]
        else:
            if current_key:
                current_value.append(line)
    
    if current_key:
        subcaptions[current_key] = ' '.join(current_value).strip()
    
    return subcaptions


def main(args: argparse.Namespace) -> None:
    """
    Main function to process captions and save results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Initialize OpenAI client
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    
    # Load dataset
    dataset = load_dataset(args.input_file)
    print(f"\nDataset size: {len(dataset)}")
    
    # Load system prompt
    with open(args.system_prompt_file, 'r') as f:
        system_prompt = f.read().strip()
    
    # Inference loop
    results = []

    for item in tqdm(dataset, desc="Processing captions", total=len(dataset), file=stderr):
        caption = item['caption']

        output = process_caption(
            client=client,
            system_prompt=system_prompt,
            caption=caption,
            model=args.model,
            max_tokens=args.max_tokens
        )
        subcaptions = parse_subcaptions(output)
        
        result = {
            'caption': caption,
            'num_subcaptions': len(subcaptions),
            'subcaptions': subcaptions,
            'output': output
        }
        results.append(result)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process captions into subcaptions")

    parser.add_argument("--input-file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-file", required=True, help="Path to output JSON file")
    parser.add_argument("--system-prompt-file", required=True, help="Path to system prompt file")
    parser.add_argument("--base-url", default="http://gpu010:8080/v1", help="Base URL for OpenAI API")
    parser.add_argument("--model", default="/model-weights/Meta-Llama-3.1-8B-Instruct", help="Model directory")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of tokens for API response")
    
    args = parser.parse_args()
    main(args)
