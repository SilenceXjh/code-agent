import json
import os
import re
from typing import Dict, List

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_mbpp_data(file_path: str) -> List[Dict]:
    """加载MBPP数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_model(model_path: str, adapter_path: str = None):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"加载 peft 模型 {adapter_path}")
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
