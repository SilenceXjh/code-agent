import os
from typing import List

from openai import OpenAI
from utils import load_jsonl_data, ds_api_generate, extract_python_code, get_testcases

data_path = "/data0/xjh/code-agent/HumanEval.jsonl"
output_path = "/data0/xjh/code-agent/human_eval/ds-test-first"


data = load_jsonl_data(data_path)

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

os.makedirs(output_path, exist_ok=True)


def get_gen_tests_prompt(prompt: str, testcases: List):
    example_test = testcases[0]

    full_prompt = f"""You are given a python programming problem and one example test case.

Your task is to generate additional valid test cases for this problem.

Requirements:
1. The test cases must follow the same format as the example test case.
2. The generated test cases should be correct according to the problem description.
3. Try to cover different cases.
4. Only output the test cases. Do not include explanations.

Problem description:
{prompt}

Example test case:
{example_test}

Generate 3-5 new test cases.
"""
    return full_prompt


for sample in data:
    task_id = sample["task_id"]
    task_id = task_id.replace('/', '_')

    prompt = sample["prompt"]
    testcases = get_testcases(sample["test"], sample["entry_point"])

    # 生成测试用例
    gen_tests_prompt = get_gen_tests_prompt(prompt, testcases)
    # print("gen tests prompt:")
    # print(gen_tests_prompt)

    generated_text = ds_api_generate(gen_tests_prompt, client)
    gen_tests = extract_python_code(generated_text)

    # print("gen tests:")
    # print(gen_tests)

    # 生成代码
    full_prompt = "Please implement the following method:\n"
    full_prompt += prompt
    full_prompt += "\nYour implementation should pass the following tests:"
    full_prompt += gen_tests
    full_prompt += "\nOnly provide the method implementation without any explanation."
    # print("full_prompt:")
    # print(full_prompt)

    generated_text = ds_api_generate(full_prompt, client)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)