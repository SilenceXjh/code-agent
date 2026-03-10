import os

from openai import OpenAI
from utils import load_jsonl_data, extract_python_code, ds_api_generate

data_path = "/data0/xjh/code-agent/HumanEval.jsonl"
output_path = "/data0/xjh/code-agent/human_eval/ds"


data = load_jsonl_data(data_path)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

os.makedirs(output_path, exist_ok=True)


for sample in data:
    task_id = sample["task_id"]
    task_id = task_id.replace('/', '_')
    prompt = sample["prompt"]
    full_prompt = "Please implement the following method:\n"
    full_prompt += prompt
    full_prompt += "\nOnly provide the method implementation without any explanation."
    # print(full_prompt)

    generated_text = ds_api_generate(prompt, client)
    # print("generated text:")
    # print(generated_text)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)