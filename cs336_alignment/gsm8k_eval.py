"""
Run GSM-8k Eval on a given model

Running:
```
python -m cs336_alignment.gsm8k_eval \
    --eval-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/gsm8k/test.jsonl" \
    --model-name "meta-llama/Meta-Llama-3-8B" \
    --output-file "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/gsm8k/eval_results"
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
from cs336_alignment.model_inference_utils import BatchPromptDispatcher
from vllm import LLM, SamplingParams
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)
HfFolder.save_token("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")

### MMLU Specific
@dataclass
class Gsm8kEvalUnit:
    question: str
    model_response: str
    model_answer: str
    ground_truth_answer: str

    
def _write_gsm8k_eval_results(eval_units: List[Gsm8kEvalUnit], output_file_path: pathlib.Path):
    all_metrics = []
    with open(output_file_path, "w") as fout:
        for eval_unit in tqdm(eval_units):
            metrics = {
                "accurate": 1.0 if eval_unit.model_answer == eval_unit.ground_truth_answer else 0.0
            }
            all_metrics.append(metrics)

            fout.write(
                json.dumps(
                    {
                        "question": eval_unit.question,
                        "model_response": eval_unit.model_response,
                        "model_answer": eval_unit.model_answer,
                        "ground_truth_answer": eval_unit.ground_truth_answer,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )

    for key in sorted(list(all_metrics[0].keys())):
        metric_mean = sum([metrics[key] for metrics in all_metrics]) / len(all_metrics)
        metric_str = f"Mean({key}): {metric_mean}\n"
        logger.info(metric_str)

def _to_gsm8k_eval_units(prompt_responses, prompt_info) -> Gsm8kEvalUnit:
    return [Gsm8kEvalUnit(question=prompt_info[prompt][0],
                         model_response=resp, 
                         model_answer=parse_gsm8k_response(resp),
                         ground_truth_answer=prompt_info[prompt][1]) for prompt, resp in prompt_responses.items()]

# Create prompts from eval set
def main(eval_file_path, model_name_or_path, output_dir):    
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
      temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    model = LLM(model=model_name_or_path)
    BATCH_SIZE = 32
    dispatcher = BatchPromptDispatcher(BATCH_SIZE, model, sampling_params)
    # dispatcher = None # DEBUG

    # Parse eval-files to generate examples, collect responses and compare with ground-truth    
    all_results: List[Gsm8kEvalUnit] = []
    
    # Prepare prompt template
    task_specific_prompt_template = construct_eval_prompt_template('gsm8k.instruction')


    # Read .jsonl file into a dataframe
    eval_examples_df = pd.read_json(eval_file_path, lines=True)
    eval_examples_df['ground_truth'] = eval_examples_df['answer'].apply(lambda x: parse_gsm8k_response(x))

    # Process dataframe: construct a prompt
    prompt_info = {} # prompt -> (subject, question, options, answer)
    if dispatcher is not None:
        for row in tqdm(eval_examples_df.itertuples(index=True, name='Gsm8kExample')):
            model_prompt = task_specific_prompt_template.format(question = row.question) 
            prompt_info[model_prompt] = (row.question, row.ground_truth)

            batch_results = dispatcher.add(model_prompt)
            if batch_results is not None:
                all_results += _to_gsm8k_eval_units(batch_results, prompt_info)
        
        # Flush after out remaining examples into a single batch
        batch_results = dispatcher.query_and_flush()
        if batch_results is not None:
            all_results += _to_gsm8k_eval_units(batch_results, prompt_info)
    
    # DEBUG: Bypass by setting dispatcher to None
    else:
        all_results = [Gsm8kEvalUnit(question=row.question, model_response=row.answer, model_answer=row.ground_truth, ground_truth_answer=row.ground_truth) for row in eval_examples_df.itertuples(index=True, name='Gsm8kExample')]
                                 
    # Write eval results to file. 
    output_file_path = pathlib.Path(output_dir) / 'eval_results.jsonl'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _write_gsm8k_eval_results(all_results, output_file_path)

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
        "--output-dir",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.eval_file,
        args.model_name,
        args.output_dir
    )
    logger.info("finished running %s", sys.argv[0])