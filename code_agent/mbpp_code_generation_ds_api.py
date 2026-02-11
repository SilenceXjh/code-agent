#!/usr/bin/env python3
"""
MBPP代码生成器
只负责生成代码，不执行测试
"""

import json
import os
from tqdm import tqdm
from openai import OpenAI

from utils.mbpp_utils import create_test_file, extract_function_code, extract_function_signature, load_mbpp_data


def generate_code(client: OpenAI, prompt: str, function_signature, max_new_tokens: int = 1024) -> str:
    """使用模型生成代码"""
    
    if function_signature:
        full_prompt = f"""Write a Python function based on the following description.

{prompt}

Use this function signature:
{function_signature}

Provide only the Python function code without any explanation."""
    else:
        full_prompt = f"""Write a Python function based on the following description.

{prompt}

Provide only the Python function code without any explanation."""
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.2,
        max_tokens=max_new_tokens,
        n=1,
        stream=False
    )
    generated_text = response.choices[0].message.content
    
    code = generated_text.strip()
    
    return full_prompt, code


def generate_all_tasks(
    data_path: str,
    output_dir: str,
    metadata_path: str,
    max_samples: int = None,
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
    client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    
    # 生成代码
    print(f"\n开始生成代码...")
    print(f"输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print()
    
    # 清空元数据文件
    # if os.path.exists(metadata_path):
    #     os.remove(metadata_path)
    
    generated_files = []
    
    for item in tqdm(data, desc="生成进度"):
        task_id = item['task_id']
        text = item['text']
        reference_code = item['code']
        test_setup_code = item.get('test_setup_code', '')
        test_list = item['test_list']
        
        # 生成代码
        function_signature = extract_function_signature(reference_code, test_list)
        full_prompt, generated_code = generate_code(client, text, function_signature)
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
            'prompt': full_prompt,
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
    DATA_PATH = os.getenv('DATA_PATH', "/data0/xjh/code-agent/mbpp.jsonl")
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "/data0/xjh/code-agent/data/deepseek-reasoner/generations")
    METADATA_PATH = os.getenv('METADATA_PATH', "/data0/xjh/code-agent/data/deepseek-reasoner/generation_metadata.jsonl")
    MAX_SAMPLES = os.getenv('MAX_SAMPLES', None)

    if MAX_SAMPLES:
        MAX_SAMPLES = int(MAX_SAMPLES)
    
    # 检查文件
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到数据文件 {DATA_PATH}")
        return
    
    # 生成代码
    generate_all_tasks(DATA_PATH, OUTPUT_DIR, METADATA_PATH, MAX_SAMPLES)


if __name__ == "__main__":
    main()