"""
Generate model predictions for a Safety Eval dataset

Running:
```
python -m cs336_alignment.simple_safety_model_predictions \
--eval-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/simple_safety_tests/simple_safety_tests.csv" \
--model-name "meta-llama/Meta-Llama-3-8B" \
--output-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/simple_safety_tests/meta-llama_3_8b_predictions.jsonl"
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
def main(eval_file_path, model_name, num_gpus, output_file):    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
      temperature=0.0, top_p=1.0, max_tokens=512
    )
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=6144,
    )
    # model = None

    # Prepare prompt template
    generic_chat_prompt_template = construct_eval_prompt_template('generic_chat.instruction')

    # Read .jsonl file into a dataframe
    eval_examples_df = pd.read_csv(eval_file_path)
    eval_responses_df = eval_examples_df.copy()
    eval_responses_df.drop(columns=['id', 'counter'])

    # Populate output dataframe
    # TODO: Figure out why even H100 takes so long to process this. Is it because we are in a DF loop?
    prompt_to_id = {}
    for row in eval_examples_df.itertuples(index=True, name='ChatExample'):
        prompt = generic_chat_prompt_template.format(instruction = row.prompts_final)
        prompt_to_id[prompt] = row.Index
    
    # Query model with full batch. The output is a list of RequestOutput objects 
    # that contain the prompt, generated text, and other information.
    prompts = [k for k in prompt_to_id.keys()]
    outputs = model.generate(prompts, sampling_params) if model is not None else ["No Model; No output" for prompt in prompts]
    model_responses = {}
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        model_responses[prompt_to_id[prompt]] = generated_text

    # Procure results
    for row in eval_examples_df.itertuples(index=True, name='SafetyExample'):
        eval_responses_df.at[row.Index, "instruction"] = row.prompts_final
        eval_responses_df.at[row.Index, "harm_area"] = row.harm_area
        eval_responses_df.at[row.Index, "category"] = row.category
        eval_responses_df.at[row.Index, "output"] = model_responses[row.Index]
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
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.eval_file,
        args.model_name,
        args.num_gpus,
        args.output_file
    )
    logger.info("finished running %s", sys.argv[0])