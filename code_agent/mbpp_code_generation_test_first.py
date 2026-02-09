#!/usr/bin/env python3
"""
MBPP代码生成器
只负责生成代码，不执行测试
"""

import json
import os
from typing import Dict, List
import torch
from tqdm import tqdm

from utils.mbpp_utils import create_test_file, extract_function_code, extract_function_signature, load_mbpp_data, load_model


def generate_tests(tokenizer, model, prompt: str, function_signature: str, test_list: List, max_new_tokens: int = 512):
    """使用模型生成测试用例"""

    example_test = test_list[0]
    
    full_prompt = f"""You are given a programming problem and one example test case.

Your task is to generate additional valid test cases for this problem.

Requirements:
1. The test cases must follow the same format as the example test case.
2. Do NOT repeat the given example test case.
3. The generated test cases should be correct according to the problem description.
4. Try to cover different cases (e.g., order differences, overlapping elements, no overlap, full overlap).
5. Only output the test cases. Do not include explanations.

Problem description:
{prompt}

Function signature:
{function_signature}

Example test case:
{example_test}

Generate 3-5 new test cases.
"""
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if full_prompt in generated_text:
        res = generated_text.split(full_prompt)[-1].strip()
    else:
        res = generated_text.strip()
    
    return full_prompt, res


def generate_code(tokenizer, model, prompt: str, gen_tests: str, max_new_tokens: int = 512):
    """使用模型生成代码"""
    
    full_prompt = f"""You are an expert Python programmer. Write a Python function based on the following description.

{prompt}

Your python function must pass the following test cases:
{gen_tests}

Provide only the Python function code without any explanation."""
    
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if full_prompt in generated_text:
        code = generated_text.split(full_prompt)[-1].strip()
    else:
        code = generated_text.strip()
    
    return full_prompt, code


def generate_all_tasks(
    model_path: str,
    data_path: str,
    output_dir: str,
    metadata_path: str,
    max_samples: int = None
):
    """
    生成所有任务的代码文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载MBPP数据集...")
    data = load_mbpp_data(data_path)
    if max_samples:
        data = data[:max_samples]
    print(f"共加载 {len(data)} 个样本")
    
    # 加载模型
    tokenizer, model = load_model(model_path)
    
    # 生成代码
    print(f"\n开始生成代码...")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print()
    
    # 清空元数据文件
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    generated_files = []
    
    for item in tqdm(data, desc="生成进度"):
        task_id = item['task_id']
        text = item['text']
        reference_code = item['code']
        test_setup_code = item.get('test_setup_code', '')
        test_list = item['test_list']
        
        # 生成代码
        function_signature = extract_function_signature(reference_code, test_list)
        test_gen_prompt, gen_tests = generate_tests(tokenizer, model, text, function_signature, test_list)
        gen_tests = extract_function_code(gen_tests)
        code_gen_prompt, generated_code = generate_code(tokenizer, model, text, gen_tests)
        extracted_code = extract_function_code(generated_code, function_signature)
        
        # 创建测试文件
        filepath = create_test_file(
            task_id,
            extracted_code,
            test_setup_code,
            test_list,
            output_dir
        )
        
        generated_files.append(filepath)
        
        # 保存元数据
        metadata = {
            'task_id': task_id,
            'test_gen_prompt': test_gen_prompt,
            'gen_tests': gen_tests,
            'code_gen_prompt': code_gen_prompt,
            'reference_code': reference_code,
            'generated_code': extracted_code,
            'test_file': filepath,
            'num_tests': len(test_list)
        }
        
        with open(metadata_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    print("\n" + "="*60)
    print("代码生成完成!")
    print(f"生成文件数: {len(generated_files)}")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print("="*60)
    
    return generated_files


def main():
    # 配置参数
    MODEL_PATH = os.getenv('MODEL_PATH', "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B")
    DATA_PATH = os.getenv('DATA_PATH', "/data0/xjh/code-agent/mbpp.jsonl")
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "/data0/xjh/code-agent/data/qwen-coder-7B-test-first/generations")
    METADATA_PATH = os.getenv('METADATA_PATH', "/data0/xjh/code-agent/data/qwen-coder-7B-test-first/generation_metadata.jsonl")
    MAX_SAMPLES = os.getenv('MAX_SAMPLES', None)
    
    if MAX_SAMPLES:
        MAX_SAMPLES = int(MAX_SAMPLES)
    
    # 检查文件
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到数据文件 {DATA_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型路径 {MODEL_PATH}")
        return
    
    # 生成代码
    generate_all_tasks(MODEL_PATH, DATA_PATH, OUTPUT_DIR, METADATA_PATH, MAX_SAMPLES)


if __name__ == "__main__":
    main()