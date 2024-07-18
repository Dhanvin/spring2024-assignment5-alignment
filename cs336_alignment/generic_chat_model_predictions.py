"""
Generate model predictions for Alpacs Eval on a given model

Running:
```
python -m cs336_alignment.generic_chat_model_predictions \
--eval-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/alpaca_eval/alpaca_eval.jsonl" \
--model-name "meta-llama/Meta-Llama-3-8B" \
--output-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/alpaca_eval/meta-llama_3_8b_predictions.jsonl"
```
"""
import pathlib
import os
import sys
import pandas as pd
import logging
import argparse
import json

from cs336_alignment.parsing_utils import parse_gsm8k_response, construct_eval_prompt_template
from vllm import LLM, SamplingParams
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)
HfFolder.save_token("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")

# Create prompts from eval set
def main(eval_file_path, model_name, output_file):    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
      temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    model = LLM(model=model_name)
    # model = None

    # Prepare prompt template
    generic_chat_prompt_template = construct_eval_prompt_template('generic_chat.instruction')

    # Read .jsonl file into a dataframe
    eval_examples_df = pd.read_json(eval_file_path, lines=True)
    eval_responses_df = eval_examples_df.copy()

    # Process dataframe: construct a prompt
    for row in tqdm(eval_examples_df.itertuples(index=True, name='ChatExample')):
        model_prompt = generic_chat_prompt_template.format(instruction = row.instruction) 
        eval_responses_df.at[row.Index, "instruction"] = row.instruction
        eval_responses_df.at[row.Index, "dataset"] = row.dataset
        eval_responses_df.at[row.Index, "output"] = model.generate(model_prompt) if model is not None else row.output
        eval_responses_df.at[row.Index, "generator"] = model_name 
    
                       
    # Write eval results to file. 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    eval_responses_df.to_json(output_file, orient='records', lines=True)


# Create a file Parser
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        type=str,
        required=True,
        help="Path to data-directory for topic-wise MMLU evaluation csv files.",
    )
    parser.add_argument(
        "--model-name", help="HF name of the model to use", required=True
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.eval_file,
        args.model_name,
        args.output_file
    )
    logger.info("finished running %s", sys.argv[0])