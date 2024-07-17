"""
Run MMLU Eval on a given model

Running:
```
python -m mmlu_eval.py \
    --eval-dir "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/mmlu/dev" \
    --model-name-or-path "/home/shared/Meta-Llama-3-70B-Instruct" \
    --output-path "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/mmlu/eval_results"
```
"""
import pathlib
import os
import sys
import random
import logging
import argparse
import json

from cs336_alignment.parsing_utils import parse_mmlu_response, mmlu_example_generator
from vllm import LLM, SamplingParams
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)
HfFolder.save_token("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")

# TODO: Use batch-inference mode. Create a simple BatchQuery class which accumulates prompts,
# dispatches to the model based on batch-size and 

@dataclass
class MmluEvalUnit:
    prompt: str
    model_response: str
    model_answer: str
    ground_truth: str

@dataclass
class MmluEvalSummary:
    subject: str
    correct: int
    incorrect: int
    sampled_failures: List[MmluEvalUnit]
    deep_dive_file_path: str
    
def generate_summary(eval_units: List[MmluEvalUnit], subject: str, deep_dive_file_path: pathlib.Path) -> MmluEvalSummary:
    #
    all_metrics = []
    with open(deep_dive_file_path, "w") as fout:
        for eval_unit in tqdm(eval_units):
            metrics = {
                "accurate": 1.0 if eval_unit.model_answer == eval_unit.ground_truth else 0.0
            }
            all_metrics.append(metrics)

            fout.write(
                json.dumps(
                    {
                        "model_prompt": eval_unit.prompt,
                        "model_response": eval_unit.model_response,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )
    for key in sorted(list(all_metrics[0].keys())):
        metric_mean = sum([metrics[key] for metrics in all_metrics]) / len(all_metrics)
        logger.info(f"Mean({key}): {metric_mean}")


class BatchPromptDispatcher():
    def __init__(self, batch_size, llm: LLM, sampling_params: SamplingParams):
        self.batch_size = batch_size
        self.prompt_responses = {}
        self.prompt_gt = {}
        self.llm = llm
        self.sampling_params = sampling_params
    
    def add(self, prompt, ground_truth):
        self.prompt_responses[prompt] = 'NOT EXECUTED'
        self.prompt_gt[prompt] = ground_truth
        if self.prompt_responses == self.batch_size:
            return self.query_model_and_flush()
        return None
    
    def query_and_flush(self) -> List[MmluEvalUnit]:
        # Query model with batch
        prompts = [k for k in self.prompt_responses.keys()]

        # Generate texts from the prompts. The output is a list of RequestOutput objects 
        # that contain the prompt, generated text, and other information.
        if self.llm is not None: 
            outputs = self.llm.generate(prompts, self.sampling_params)
            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text.strip()
                self.prompt_responses[prompt] = generated_text

        ### DEBUG ONY
        # The expectation is that the model generates the answer, 
        # closes the markdown code block (with ```), 
        # and then starts the next conversation turn (with # Query:). 
        # Thus, when we see the string # Query: we can stop response generation.
        else: 
            for prompt in self.prompt_responses.keys():
                self.prompt_responses[prompt] = prompt + f"The correct answer is {self.prompt_gt[prompt]}.```"

        # Procure results
        output = [MmluEvalUnit(prompt, resp, parse_mmlu_response(resp), self.prompt_gt[prompt]) for prompt, resp in self.prompt_responses.items()]
        self.prompt_responses = {}
        self.prompt_gt = {}
        return output


def _construct_mmlu_eval_prompt_template():
    # Get System prompt
    prompt_dir = pathlib.Path(__file__).parent /'prompts'
    system_prompt_file = prompt_dir / 'zero_shot_system_prompt.prompt'
    with open(system_prompt_file, 'r') as f:
        system_prompt_template = f.read()


    # Get Eval instruction for system prompt
    instruction_file = prompt_dir / 'mmlu_eval.instruction'
    with open(instruction_file, 'r') as f:
        instruction_prompt = f.read()

    task_specific_prompt_template = system_prompt_template.format(instruction=instruction_prompt)
    return task_specific_prompt_template

# Create prompts from eval set
def main(eval_dir, model_name_or_path, output_dir_str):
    eval_dir_path = pathlib.Path(eval_dir)
    
    model = LLM(model=model_name_or_path)
    # model = None
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
      temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    task_specific_prompt_template = _construct_mmlu_eval_prompt_template()
    N_FILES = 20
    random.seed(42)
    # eval_dir_path = pathlib.Path(__file__).parent.parent / 'data' / 'mmlu' / 'dev'
    eval_files = os.listdir(eval_dir_path)
    def get_subject_from_filename(filename: str):
        return '_'.join(pathlib.Path(filename).stem.split('_')[:-1])
    files = random.sample(eval_files, min(N_FILES, len(eval_files)))

    BATCH_SIZE = 32
    dispatcher = BatchPromptDispatcher(BATCH_SIZE, model, sampling_params)
    all_results = []
    all_results = []
    for file_name in tqdm(files):
        file_path = eval_dir_path / file_name
        subject_str = get_subject_from_filename(file_path.stem)
        print(f"Processing {file_path}")
        with open(file_path, 'r') as f:
            for parsed_example in mmlu_example_generator(f):
                model_prompt = task_specific_prompt_template.format(subject = subject_str,
                                                                    question = parsed_example['question'],
                                                                    options = parsed_example['options']) 
                batch_results = dispatcher.add(model_prompt, parsed_example['answer'])
                if batch_results is not None:
                    all_results += batch_results
            
            # Don't mix batches across subjects. Flush after each subject
            batch_results = dispatcher.query_and_flush()
            if batch_results is not None:
                all_results += batch_results
                    
                    
        # Write eval results to file. 
        # Generate a summary.txt file for a) overall summary and b) per-subject stats
        output_dir_path = pathlib.Path(output_dir_str) / model_name_or_path / eval_dir_path.stem
        # Create a dir for the model
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        # Create JSONL path
        file_path = output_dir_path / file_name
        deep_dive_file_name = str(file_path.stem) + '.jsonl'
        generate_summary(all_results, subject_str, deep_dive_file_path = output_dir_path / deep_dive_file_name )


# Create a file Parser
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Path to data-directory for topic-wise MMLU evaluation csv files.",
    )
    parser.add_argument(
        "--model-name", help="HF name of the model to use", required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.eval_dir,
        args.model_name,
        args.output_path
    )
    logger.info("finished running %s", sys.argv[0])