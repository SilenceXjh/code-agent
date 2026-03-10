import os
from utils import load_jsonl_data, load_model, model_generate, extract_python_code, extract_function_signature

data_path = "/data0/xjh/code-agent/mbpp.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
output_path = "/data0/xjh/code-agent/mbpp/qwen-1.5B"


data = load_jsonl_data(data_path)
tokenizer, model = load_model(model_path)

os.makedirs(output_path, exist_ok=True)


for sample in data:
    task_id = sample["task_id"]
    func_desc = sample["text"]
    function_signature = extract_function_signature(sample["code"], sample["test_list"])
    full_prompt = f"""Write a Python function based on the following description.

{func_desc}

Use this function signature:
{function_signature}

Provide only the Python function code without any explanation."""
    # print(full_prompt)

    generated_text = model_generate(full_prompt, model, tokenizer)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)