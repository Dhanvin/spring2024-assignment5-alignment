from vllm import LLM, SamplingParams

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
    
    def query_and_flush(self):
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
