import pandas as pd

def process_mmlu_eval_results(file_path):
    """
    Approach
        Extract the 'accurate' metric for each subject
        This will create a multi-indexed pd.Series, allowing us to further group by 'subject'
            - Primary index is named: 'subject' and secondary index is the original index (row-id)
        Calculate the average accuracy for each subject from the multi-indexed Series
    """
    df = pd.read_json(file_path, lines=True) # since it is a .jsonl file
    accuracy_df = df.groupby('subject')['metrics'].apply(lambda x: x.apply(lambda y: y['accurate']))
    average_accuracy = accuracy_df.groupby('subject').mean().reset_index()
    average_accuracy.columns = ['subject', 'average_accuracy']
    print(average_accuracy)

# Load the .jsonl file into a list of dictionaries
# process_mmlu_eval_results('/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/mmlu/meta-llama/Meta-Llama-3-8B/dev/eval_results.jsonl')


file_path_vanilla_prompt = "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/gsm8k/eval_results/eval_results.jsonl"
file_path_crafted_prompt = "/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/gsm8k/eval_results/eval_results_single_sentence.jsonl"
df_crafted_prompt = pd.read_json(file_path_crafted_prompt, lines=True) # since it is a .jsonl file
df_vanilla_prompt = pd.read_json(file_path_vanilla_prompt, lines=True) # since it is a .jsonl file

# Merge DataFrames on 'question'
merged_df = pd.merge(df_crafted_prompt, df_vanilla_prompt, on='question', suffixes=('_crafted', '_vanilla'))
merged_df = merged_df.drop(columns=['ground_truth_answer_vanilla', 'metrics_crafted', 'metrics_vanilla'])
merged_df = merged_df.rename(columns={'ground_truth_answer_crafted': 'gt_answer'})

# Group 1: df1's model_answer matches ground_truth_answer but df2's doesn't
crafted_better_group = merged_df[
    (merged_df['model_answer_crafted'] == merged_df['gt_answer']) & 
    (merged_df['model_answer_vanilla'] != merged_df['gt_answer'])
]

# Group 2: df1's model_answer doesn't match ground_truth_answer but df2's does
vanilla_better_group = merged_df[
    (merged_df['model_answer_crafted'] != merged_df['gt_answer']) & 
    (merged_df['model_answer_vanilla'] == merged_df['gt_answer'])
]

print("Group 1: crafted Better")
print(crafted_better_group)

print("\nGroup 2: Vanilla Better")
print(vanilla_better_group)
# breakpoint()