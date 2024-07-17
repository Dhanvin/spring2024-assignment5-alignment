import pandas as pd

def process_eval_results(file_path):
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
process_eval_results('/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment5-alignment/data/mmlu/meta-llama/Meta-Llama-3-8B/dev/eval_results.jsonl')