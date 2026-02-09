#!/usr/bin/env python3
"""
MBPP代码生成器
只负责生成代码，不执行测试
"""

import json
import os
from typing import Dict, List
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re


def load_mbpp_data(file_path: str) -> List[Dict]:
    """加载MBPP数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_model(model_path: str):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成!")
    return tokenizer, model


def extract_function_name_from_tests(test_list: List):
    """从测试用例中提取函数名"""
    if not test_list:
        return None
    
    # 取第一个测试用例
    first_test = test_list[0]
    
    # 匹配 assert function_name(
    match = re.search(r'assert\s+(\w+)\s*\(', first_test)
    if match:
        return match.group(1)
    
    return ""

def extract_function_signature(reference_code: str, test_list: List) -> str:
    """从参考代码中提取函数签名"""
    func_name = extract_function_name_from_tests(test_list)
    func_name = func_name.strip()
    lines = reference_code.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('def ') and func_name in line:
            return line
    return ""

def generate_code(tokenizer, model, prompt: str, function_signature, max_new_tokens: int = 1024) -> str:
    """使用模型生成代码（带思考过程）"""
    
    system_prompt = "You are an expert Python programmer."

    if function_signature:
        full_prompt = f"""Write a Python function based on the following description.

{prompt}

Use this function signature:
{function_signature}

Before writing the code, think through these steps:
1. What are the key requirements?
2. What edge cases should be handled?
3. What's the best approach to solve this?

Format your response as:
<thinking>
[Your analysis here]
</thinking>

<code>
[Your Python function code here]
</code>"""
    else:
        full_prompt = f"""Write a Python function based on the following description.

{prompt}

Before writing the code, think through these steps:
1. What are the key requirements?
2. What edge cases should be handled?
3. What's the best approach to solve this?

Format your response as:
<thinking>
[Your analysis here]
</thinking>

<code>
[Your Python function code here]
</code>"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    #print("model input text:", text)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # 增加token数以容纳思考过程
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取思考过程和代码
    if full_prompt in generated_text:
        response = generated_text.split(full_prompt)[-1].strip()
    else:
        response = generated_text.strip()
    
    # 解析思考过程和代码
    thinking = ""
    code = response
    
    if "<thinking>" in response and "</thinking>" in response:
        thinking = response.split("<thinking>")[1].split("</thinking>")[0].strip()
    
    if "<code>" in response and "</code>" in response:
        code = response.split("<code>")[1].split("</code>")[0].strip()
    elif "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    
    return full_prompt, thinking, code


def extract_function_code(generated_text: str, expected_signature: str = "") -> str:
    """从生成的文本中提取函数代码"""
    if "```python" in generated_text:
        code = generated_text.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_text:
        code = generated_text.split("```")[1].split("```")[0].strip()
    else:
        code = generated_text.strip()
    
    if expected_signature:
        import re
        expected_func_match = re.search(r'def\s+(\w+)\s*\(', expected_signature)
        if expected_func_match:
            expected_func_name = expected_func_match.group(1)
            if f"def {expected_func_name}" not in code:
                generated_func_match = re.search(r'def\s+\w+\s*\([^)]*\):', code)
                if generated_func_match:
                    old_signature = generated_func_match.group(0)
                    code = code.replace(old_signature, expected_signature + ':', 1)
    
    return code


def create_test_file(
    task_id: int,
    generated_code: str,
    test_setup_code: str,
    test_cases: List[str],
    output_dir: str
) -> str:
    """
    创建包含生成代码和测试用例的Python文件
    """
    # 创建文件名
    filename = f"task_{task_id:04d}.py"
    filepath = os.path.join(output_dir, filename)
    
    # 构建完整的测试文件
    file_content = f'''"""
MBPP Task {task_id}
Auto-generated test file
"""

# Setup code (if any)
{test_setup_code if test_setup_code else "# No setup code"}

# Generated code
{generated_code}

# Test cases
if __name__ == "__main__":
    try:
{chr(10).join("        " + test for test in test_cases)}
        print("ALL_TESTS_PASSED")
    except AssertionError as e:
        print(f"ASSERTION_FAILED: {{e}}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {{type(e).__name__}}: {{e}}")
        exit(1)
'''
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    return filepath


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
        full_prompt, thinking, generated_code = generate_code(tokenizer, model, text, function_signature)
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
            'thinking': thinking,
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
    MODEL_PATH = os.getenv('MODEL_PATH', "/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct")
    DATA_PATH = os.getenv('DATA_PATH', "/data0/xjh/code-agent/mbpp.jsonl")
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "/data0/xjh/code-agent/data/qwen-coder-7B-cot/generations")
    METADATA_PATH = os.getenv('METADATA_PATH', "/data0/xjh/code-agent/data/qwen-coder-7B-cot/generation_metadata.jsonl")
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