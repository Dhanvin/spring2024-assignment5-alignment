from typing import Any, Dict, Tuple, Iterator, List
import regex as re

def parse_mmlu_response(
    model_output: str,
) -> str | None:
    """
    Assumes that the model is prompted to provide a response where the first sentence is:
    "The correct answer is _"
    
    Input:
        model_output: str
    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    _LEN_OF_INTEREST = 30
    # Extract first sentence and associated words
    first_sentence = model_output[:_LEN_OF_INTEREST].strip().split('.')[0].strip()
    words = first_sentence.split(' ')
    if len(words) != 5:
        return None
    if words[0:4] != ['The', 'correct', 'answer', 'is']:
        return None
    ans = words[4]
    if ans in {"A", "B", "C", "D"}:
        return ans
    else:
        return None

def parse_mmlu_example(
    example: str,
) -> Dict[str, Any]:
    """
    See data/mmlu/dev/abstract_algebra_dev.csv for format. 
    Note that , within "" are to be treated as nesting statements
    Expect 5 parts, where last part will be 
    """
    example = example.strip()
    start_idx = 0
    process_until_idx = len(example) - 1
    
    parts = [] # question --> 4 options --> answer
    while start_idx < process_until_idx:
        end_pattern = ','
        pattern_prefix_len = 0
        if example[start_idx] == '"':
            end_pattern = '",'
            pattern_prefix_len = 1
        # for pattern in prioritized_split_patterns:
        match_idx = example[start_idx:].find(end_pattern)
        if match_idx > 0:
            extracted_str = example[start_idx + pattern_prefix_len: start_idx + match_idx]
            parts.append(extracted_str)
            start_idx += match_idx + len(end_pattern) # start after end_pattern

    assert start_idx == process_until_idx, f"Example: {example}\nstart_idx: {start_idx} != process_until_idx: {process_until_idx}"
    return {
        'question': parts[0],
        'options': parts[1:],
        'answer': example[start_idx], # Last character
    }

    
def mmlu_example_generator(fp, chunk_size=8192) -> Iterator[Dict]:
    buffer = ""
    read_size = chunk_size # Adaptive. We will increase if the example doesn't fit.

    # Escape the special characters in each pattern and join with '|'
    example_ends = re.compile('|'.join(re.escape(p) for p in set([',A\n', ',B\n', ',C\n', ',D\n'])))
    total_example_cnt = 0
    while True:
        try:
            buffer += fp.read(read_size)
            buffer_local_offset = 0
            if not buffer:
                break
            # Process buffer until half the buffer is read, continue. Assumes sufficient chunk-size
            examples_extracted_within_buffer = 0
            for match in example_ends.finditer(buffer):
                yield parse_mmlu_example(buffer[buffer_local_offset: match.end()])
                buffer_local_offset = match.end()
                examples_extracted_within_buffer += 1
            total_example_cnt += examples_extracted_within_buffer
            buffer =  buffer[buffer_local_offset:] # Remove portion of buffer that's already read
            
            # Give up. Stop parsing if we are unable to extract
            if examples_extracted_within_buffer == 0:
                # read_size *= 2 # We need to increase read size
                break
                        
        except StopIteration:
            # We have reached the end of file
            print(f"End of file reached. Found {total_example_cnt} examples. Remaining buffer len: {len(buffer[buffer_local_offset:])}")
            break

