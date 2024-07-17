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
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

from vllm import LLM, SamplingParams
from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)
HfFolder.save_token("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")

@dataclass
class MmluEvalUnit:
    subject: str
    question: str
    options: str
    model_response: str
    model_answer: str
    ground_truth_answer: str

    
def write_results(eval_units: List[MmluEvalUnit], subject: str, output_file_path: pathlib.Path):
    all_metrics = []
    with open(output_file_path, "w") as fout:
        for eval_unit in tqdm(eval_units):
            metrics = {
                "accurate": 1.0 if eval_unit.model_answer == eval_unit.ground_truth else 0.0
            }
            all_metrics.append(metrics)

            fout.write(
                json.dumps(
                    {
                        "subject": eval_unit.subject,
                        "question": eval_unit.question,
                        "options": eval_unit.options,
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


class BatchPromptDispatcher():
    def __init__(self, batch_size, llm: LLM, sampling_params: SamplingParams):
        self.batch_size = batch_size
        self.prompt_responses = {}
        self.llm = llm
        self.sampling_params = sampling_params
    
    def add(self, prompt, ground_truth):
        self.prompt_responses[prompt] = 'NOT EXECUTED'
        if self.prompt_responses == self.batch_size:
            return self.query_model_and_flush()
        return None
    
    def query_and_flush(self) -> List[MmluEvalUnit]:
        # Query model with batch
        prompts = [k for k in self.prompt_responses.keys()]

        # Generate texts from the prompts. The output is a list of RequestOutput objects 
        # that contain the prompt, generated text, and other information.
        outputs = self.llm.generate(prompts, self.sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            self.prompt_responses[prompt] = generated_text

        # Procure results
        output = self.prompt_responses
        self.prompt_responses = {}
        return output

def _to_mmlu_eval_units(prompt_responses, prompt_info) -> MmluEvalUnit:
    return [MmluEvalUnit(subject=prompt_info[prompt][0],
                         question=prompt_info[prompt][1],
                         options=prompt_info[prompt][2],
                         model_response=resp, 
                         model_answer=parse_mmlu_response(resp),
                         ground_truth_answer=prompt_info[prompt][3]) for prompt, resp in prompt_responses.items()]

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
    eval_files = os.listdir(eval_dir_path)
    
    ## Samlpe a subset of files for testing
    # N_FILES = 20
    # random.seed(42)
    # files = random.sample(eval_files, min(N_FILES, len(eval_files)))

    def _get_subject_from_filename(filename: str):
        return '_'.join(pathlib.Path(filename).stem.split('_')[:-1])
    
    # Parse eval-files to generate examples, collect responses and compare with ground-truth    
    BATCH_SIZE = 32
    dispatcher = BatchPromptDispatcher(BATCH_SIZE, model, sampling_params)
    all_results: List[MmluEvalUnit] = []
    for file_name in tqdm(eval_files):
        prompt_info = {} # prompt -> (subject, question, options, answer)
        file_path = eval_dir_path / file_name
        subject_str = _get_subject_from_filename(file_path.stem)
        print(f"Processing {file_path}")
        with open(file_path, 'r') as f:
            for parsed_example in mmlu_example_generator(f):
                model_prompt = task_specific_prompt_template.format(subject = subject_str,
                                                                    question = parsed_example['question'],
                                                                    options = parsed_example['options']) 
                prompt_info[model_prompt] = (subject_str, parsed_example['question'], parsed_example['options'], parsed_example['answer'])
                batch_results = dispatcher.add(model_prompt)
                if batch_results is not None:
                    all_results += _to_mmlu_eval_units(batch_results, prompt_info)
            
            # Don't mix batches across subjects. Flush after each subject
            batch_results = dispatcher.query_and_flush()
            if batch_results is not None:
                all_results += _to_mmlu_eval_units(batch_results, prompt_info)
                    
                    
        # Write eval results to file. 
        output_dir_path = pathlib.Path(output_dir_str) / model_name_or_path / eval_dir_path.stem
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        output_file_path = output_dir_path / 'eval_results.jsonl'
        write_results(all_results, subject_str, output_file_path)

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