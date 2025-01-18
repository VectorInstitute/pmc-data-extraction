import argparse
import re
from sys import stderr
from typing import Dict

from openai import OpenAI
from tqdm import tqdm

from openpmcvl.granular.pipeline.utils import load_dataset, save_jsonl


PROMPT = """
Subfigure labels are letters referring to individual subfigures within a larger figure.
Check if the caption contains explicit subfigure label.
If not, output "NO" and end the generation.
If yes, output "YES", then generate the subcaption of the subfigures according to the caption.
The output should use the template:
    YES
    Subfigure-A: ...
    Subfigure-B: ...
    ...
The label should be removed from subcaption.
""".strip()


def process_caption(
    client: OpenAI, system_prompt: str, caption: str, model: str, max_tokens: int
) -> str:
    """
    Process a caption using the language model.

    Args:
        client (OpenAI): OpenAI client instance.
        system_prompt (str): System prompt for the language model.
        caption (str): Caption to process.
        model (str): Model directory being used.
        max_tokens (int): Maximum number of tokens for the model response.

    Returns
    -------
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

    Returns
    -------
        Dict[str, str]: Dictionary of subcaptions, where keys are subfigure labels and values are subcaptions.
    """
    lines = output.strip().split("\n")

    if not lines[0].upper().startswith("YES"):
        return {"Subfigure-A": "\n".join(lines)}

    subcaptions = {}
    current_key = None
    current_value = []

    for line in lines[1:]:  # Skip the "YES" line
        match = re.match(r"^Subfigure-([A-Za-z]):\s*(.*)", line, re.IGNORECASE)

        if match:
            if current_key:
                subcaptions[current_key] = " ".join(current_value).strip()
            current_key = f"Subfigure-{match.group(1).upper()}"
            current_value = [match.group(2)]
        elif current_key:
            current_value.append(line)

    if current_key:
        subcaptions[current_key] = " ".join(current_value).strip()

    return subcaptions


def main(args: argparse.Namespace) -> None:
    """
    Main function to process captions and save results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Initialize OpenAI client
    client = OpenAI()  # base_url=args.base_url, api_key="EMPTY"

    # Load dataset
    dataset = load_dataset(args.input_file)
    print(f"\nDataset size: {len(dataset)}")

    # Inference loop
    results = []

    for item in tqdm(
        dataset, desc="Processing captions", total=len(dataset), file=stderr
    ):
        caption = item["caption"]

        output = process_caption(
            client=client,
            system_prompt=PROMPT,
            caption=caption,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        subcaptions = parse_subcaptions(output)

        item["num_subcaptions"] = len(subcaptions)
        item["subcaptions"] = subcaptions
        item["llm_output"] = output

        results.append(item)

    save_jsonl(results, args.output_file)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process captions into subcaptions")

    parser.add_argument("--input-file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-file", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens for API response",
    )

    args = parser.parse_args()
    main(args)
