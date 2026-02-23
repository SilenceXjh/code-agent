import json
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载jsonl格式的数据集"""
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


def model_generate(prompt: str, model, tokenizer, is_instruct=True, max_new_tokens=512):
    if is_instruct:
        messages = [
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print("model input text:", text)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
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
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[-1].strip()
        
        # print("model generated text:", generated_text)
        return generated_text  
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
        
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[-1].strip()

        return generated_text
    

def extract_python_code(generated_text: str) -> str:
    """从生成的文本中提取函数代码"""
    if "```python" in generated_text:
        code = generated_text.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_text:
        code = generated_text.split("```")[1].split("```")[0].strip()
    else:
        code = generated_text.strip()
    
    return code


def get_testcases(orinigal_test: str, entry_point: str):
    testcases = []
    test_code_lines = orinigal_test.split("\n")
    for line in test_code_lines:
        line = line.strip()
        if "assert" in line:
            testcase = line.replace("candidate", entry_point)
            testcases.append(testcase)

    return testcases


def construct_file_content(code_str: str, testcases: List):
    test_script = f"""
# Generated code
{code_str}

# 测试用例
def run_tests():
    test_cases = {repr(testcases)}
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            exec(test_case)
            passed += 1
            
        except AssertionError as e:
            failed += 1
            print(f"测试失败: {{test_case}}")
            print(f"AssertionError: {{e if str(e) else '断言失败'}}")
            
        except Exception as e:
            failed += 1
            print(f"测试失败: {{test_case}}")
            print(f"{{type(e).__name__}}: {{e}}")
    
    if failed == 0:
        print("ALL_TESTS_PASSED")
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        exit(run_tests())
    except KeyboardInterrupt:
        print("\\n测试被中断")
        sys.exit(1)
"""
    # print(test_script)
    return test_script