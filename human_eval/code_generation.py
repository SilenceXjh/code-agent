import os
from utils import load_jsonl_data, load_model, model_generate, extract_python_code

data_path = "/data0/xjh/code-agent/HumanEval.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_path = "/data0/xjh/code-agent/human_eval/qwen-7B"


data = load_jsonl_data(data_path)
tokenizer, model = load_model(model_path)

os.makedirs(output_path, exist_ok=True)


for sample in data:
    task_id = sample["task_id"]
    task_id = task_id.replace('/', '_')
    prompt = sample["prompt"]
    full_prompt = "Please implement the following method:\n"
    full_prompt += prompt
    full_prompt += "\nOnly provide the method implementation without any explanation."
    print(full_prompt)

    generated_text = model_generate(full_prompt, model, tokenizer)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)