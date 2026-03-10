import os

from openai import OpenAI
from utils import load_jsonl_data, ds_api_generate, extract_python_code, extract_function_signature

data_path = "/data0/xjh/code-agent/mbpp.jsonl"
output_path = "/data0/xjh/code-agent/mbpp/ds"


data = load_jsonl_data(data_path)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

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

    generated_text = ds_api_generate(full_prompt, client)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)