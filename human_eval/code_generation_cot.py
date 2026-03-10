import os
from utils import load_jsonl_data, load_model, model_generate, extract_python_code

data_path = "/data0/xjh/code-agent/HumanEval.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
output_path = "/data0/xjh/code-agent/human_eval/qwen-1.5B-cot"


data = load_jsonl_data(data_path)
tokenizer, model = load_model(model_path)

os.makedirs(output_path, exist_ok=True)


for sample in data:
    task_id = sample["task_id"]
    task_id = task_id.replace('/', '_')
    prompt = sample["prompt"]
    full_prompt = "Please implement the following method:\n"
    full_prompt += prompt
    full_prompt += """\nBefore generating the code, think through the problem step by step, identify the key requirements, edge cases and correct algorithm.

Then output the final code in the following format:
```python
[Your code]
```
"""
    #print(full_prompt)

    generated_text = model_generate(full_prompt, model, tokenizer)
    #print("generated text:", generated_text)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)