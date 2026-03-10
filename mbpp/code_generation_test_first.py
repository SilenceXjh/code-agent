import os
from typing import List
from utils import load_jsonl_data, load_model, model_generate, extract_python_code, extract_function_signature

data_path = "/data0/xjh/code-agent/mbpp.jsonl"
model_path = "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct/"
output_path = "/data0/xjh/code-agent/mbpp/qwen-7B-test-first-1"


data = load_jsonl_data(data_path)
tokenizer, model = load_model(model_path)

os.makedirs(output_path, exist_ok=True)

def get_gen_tests_prompt(func_desc: str, testcases: List, function_signature: str):
    example_test = testcases[0]

    full_prompt = f"""You are given a python programming problem and one example test case.

Your task is to generate additional valid test cases for this problem.

Requirements:
1. The test cases must follow the same format as the example test case.
2. The generated test cases should be correct according to the problem description.
3. Try to cover different cases.
4. Only output the test cases. Do not include explanations.

Problem description:
{func_desc}

Function signature:
{function_signature}

Example test case:
{example_test}

Generate 3-5 new test cases.
"""
    return full_prompt


for sample in data[34:35]:
    task_id = sample["task_id"]
    func_desc = sample["text"]
    testcases = sample["test_list"]
    function_signature = extract_function_signature(sample["code"], sample["test_list"])

    # 生成测试
    gen_test_prompt = get_gen_tests_prompt(func_desc, testcases, function_signature)
    generated_text = model_generate(gen_test_prompt, model, tokenizer)
    gen_tests = extract_python_code(generated_text)
    print("gen_tests:")
    print(gen_tests)

    # 生成代码
    full_prompt = f"""Write a Python function based on the following description.

{func_desc}

Use this function signature:
{function_signature}

Your implementation should pass the following tests:
{gen_tests}

Provide only the Python function code without any explanation."""
    print("full prompt:")
    print(full_prompt)

    generated_text = model_generate(full_prompt, model, tokenizer)
    code = extract_python_code(generated_text)

    with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
        f.write(code)