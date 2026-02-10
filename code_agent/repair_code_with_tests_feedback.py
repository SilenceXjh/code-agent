import subprocess
import json
from typing import List, Dict, Tuple, Optional
import torch
import os
import time

from tqdm import tqdm
from utils.mbpp_utils import extract_function_code, load_mbpp_data, load_model, load_metadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DockerTestRunner:
    def __init__(self, test_dir: str, image: str = "python:3.10-slim"):
        """
        初始化 Docker 测试运行器
        
        Args:
            test_dir: 测试文件所在目录（将被挂载到容器）
            image: Docker 镜像名称
        """
        self.test_dir = os.path.abspath(test_dir)
        self.image = image
        self.container_id = None
        
        # 确保测试目录存在
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 启动容器
        self._start_container()
    
    def _start_container(self):
        """启动一个长期运行的容器"""
        try:
            self.container_id = subprocess.check_output([
                'docker', 'run', '-d', '--rm',
                '--network', 'none',
                '--memory', '512m',  # 内存限制
                '--cpus', '1.0',     # CPU 限制
                '-v', f'{self.test_dir}:/app',
                '-w', '/app',
                self.image,
                'tail', '-f', '/dev/null'  # 保持容器运行
            ]).decode().strip()
            print(f"启动 Docker 容器: {self.container_id[:12]}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"启动 Docker 容器失败: {e}")
    
    def run_test(self, test_file: str, timeout: int = 5) -> Tuple[bool, str, float]:
        """
        在已有容器中执行测试
        
        Args:
            test_file: 测试文件名（相对于 test_dir）
            timeout: 超时时间（秒）
            
        Returns:
            (passed, error_msg, elapsed_time)
        """
        if self.container_id is None:
            raise RuntimeError("容器未启动")
        
        start_time = time.time()
        
        try:
            # 使用 timeout 命令在容器内控制超时
            # timeout 的返回码：124 表示超时，其他表示命令本身的退出码
            result = subprocess.run(
                ['docker', 'exec', 
                 self.container_id, 
                 'timeout', '--signal=KILL', f'{timeout}s',  # 使用 KILL 信号确保进程被终止
                 'python3', test_file],
                capture_output=True, 
                text=True, 
                timeout=timeout + 3  # 外层超时时间稍长一些
            )
            
            elapsed_time = time.time() - start_time
            
            # 检查退出码
            # 124: timeout 命令的超时退出码
            # 137: SIGKILL (128 + 9)
            # 143: SIGTERM (128 + 15)
            if result.returncode == 124 or result.returncode == 137:
                return False, f"TIMEOUT: 代码执行超过 {timeout} 秒（可能存在死循环）", elapsed_time
            elif result.returncode == 0:
                return True, "", elapsed_time
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                error_msg = error_msg if error_msg else "Unknown error"
                return False, error_msg, elapsed_time
                
        except subprocess.TimeoutExpired:
            # 外层超时（备用保护）
            elapsed_time = time.time() - start_time
            # 尝试杀死容器内的进程
            try:
                subprocess.run(
                    ['docker', 'exec', self.container_id, 'pkill', '-9', 'python3'],
                    timeout=2,
                    capture_output=True
                )
            except:
                pass
            return False, f"TIMEOUT: 执行超过 {timeout + 3} 秒（外层超时保护）", elapsed_time
        except Exception as e:
            elapsed_time = time.time() - start_time
            return False, f"执行错误: {str(e)}", elapsed_time
    
    def cleanup(self):
        """清理容器"""
        if self.container_id is None:
            return
        
        try:
            print(f"正在停止 Docker 容器...")
            subprocess.run(
                ['docker', 'stop', self.container_id],
                timeout=15,
                capture_output=True
            )
            print("Docker 容器已停止")
            self.container_id = None
        except Exception as e:
            print(f"停止容器时出错: {e}")


class CodeFixWorkflow:
    def __init__(self, model_path: str, 
                 output_dir: str,
                 max_iterations: int = 5,
                 docker_image: str = "python:3.10-slim",
                 test_timeout: int = 5):
        """
        初始化代码修复工作流
        
        Args:
            model_path: 本地模型路径或HuggingFace模型名
            max_iterations: 最大修复迭代次数
            docker_image: Docker镜像名称
            test_timeout: 单个测试用例的超时时间（秒）
        """
        
        self.tokenizer, self.model = load_model(model_path)
        self.device = "cuda"
        self.max_iterations = max_iterations
        self.test_timeout = test_timeout
        self.output_dir = output_dir
        
        self.docker_runner = DockerTestRunner(output_dir, docker_image)
    
    def _create_test_file(self, code: str, test_cases: List[str], 
                          test_file_path: str, test_setup_code: str = ""):
        """
        创建测试文件
        
        Args:
            code: 要测试的代码
            test_cases: 测试用例列表
            test_file_path: 测试文件路径
        """
        
        test_script = f"""
# Setup code (if any)
{test_setup_code if test_setup_code else "# No setup code"}       

# Generated code
{code}

# 测试用例
def run_tests():
    test_cases = {repr(test_cases)}
    
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
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
    
    def run_test_cases(self, code: str, test_cases: List[str], task_id, iteration: int):
        """
        在 Docker 容器中运行测试用例
        
        Args:
            code: 要测试的代码
            test_cases: 测试用例列表
            
        Returns:
            (all_passed, error_message): 是否全部通过和详细结果
        """
        if self.docker_runner is None:
            raise RuntimeError("Docker 容器未启动")
        
        # 创建测试文件
        test_file = f'task_{task_id}_test_code_{iteration}.py'
        test_file_path = os.path.join(self.output_dir, test_file)
        self._create_test_file(code, test_cases, test_file_path)
        
        # 运行测试
        passed, error_msg, elapsed_time = self.docker_runner.run_test(
            test_file, 
            timeout=self.test_timeout
        )
        
        # 解析结果
        return passed, error_msg
    
    def fix_code_with_llm(self, prompt: str, original_code: str, test_feedback: str, 
                          iteration: int):
        """
        使用本地LLM修复代码
        
        Args:
            prompt: 问题描述
            original_code: 当前的代码
            test_feedback: 测试反馈信息
            iteration: 当前迭代次数
            
        Returns:
            修复后的代码
        """
        full_prompt = f"""You are an expert Python programmer. Please fix a function base on the task description, current code and test feedback.

Task Description:
{prompt}
        
Current code:
```python
{original_code}
```

Test feedBack: 
{test_feedback}

Provide only the fixed Python function code without any explanation."""
        
        # 生成
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if full_prompt in generated_text:
            code = generated_text.split(full_prompt)[-1].strip()
        else:
            code = generated_text.strip()
        
        code = extract_function_code(code)
        
        return full_prompt, code
    
    
    def run_workflow(self, initial_code: str, test_cases: List[str], task_id, prompt: str) -> Dict:
        """
        运行完整的修复工作流
        
        Args:
            initial_code: 初始代码
            test_cases: 测试用例列表 (assert 语句)
            task_id: task_id
            prompt: 问题描述
            
        Returns:
            包含修复历史和最终结果的字典
        """
        # print("=" * 60)
        # print("开始代码自动修复工作流")
        # print("=" * 60)
            
        history = []
        current_code = initial_code
        fix_prompt = ""
        
        for iteration in range(self.max_iterations):
            
            # 运行测试
            all_passed, error_message = self.run_test_cases(current_code, test_cases, task_id, iteration)
            
            # 记录历史
            iteration_record = {
                'iteration': iteration,
                'code': current_code,
                "fix_prompt": fix_prompt,
                'all_passed': all_passed,
                'error_message': error_message
            }
            history.append(iteration_record)
            
            if all_passed:
                return {
                    'success': True,
                    'final_code': current_code,
                    'iterations': iteration,
                    'history': history
                }
            
            #尝试修复
            try:
                fix_prompt, current_code = self.fix_code_with_llm(
                    prompt,
                    current_code, 
                    error_message, 
                    iteration
                )
            except Exception as e:
                print(f"❌ LLM修复失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': f'LLM修复失败: {str(e)}',
                    'final_code': current_code,
                    'iterations': iteration + 1,
                    'history': history
                }
        
        all_passed, error_message = self.run_test_cases(current_code, test_cases, task_id, iteration)
        iteration_record = {
            'iteration': iteration,
            'code': current_code,
            'all_passed': all_passed,
            'error_message': error_message
        }
        history.append(iteration_record)
        
        if all_passed:
            return {
                'success': True,
                'final_code': current_code,
                'iterations': iteration,
                'history': history
            }

        return {
            'success': False,
            'error': f'达到最大迭代次数',
            'final_code': current_code,
            'iterations': self.max_iterations,
            'history': history
        }
        
    def cleanup(self):
        self.docker_runner.cleanup()


def main():
    MBPP_DATA_PATH = "/data0/xjh/code-agent/mbpp.jsonl"
    GENERAION_METADATA_PATH = "/data0/xjh/code-agent/data/qwen-coder-7B-test-first/generation_metadata.jsonl"
    TEST_RESULT_PATH = "/data0/xjh/code-agent/data/qwen-coder-7B-test-first/test_results.jsonl"
    mbpp_data = load_mbpp_data(MBPP_DATA_PATH)
    gen_metadata = load_metadata(GENERAION_METADATA_PATH)
    test_results = load_metadata(TEST_RESULT_PATH)

    OUTPUT_DIR = "/data0/xjh/code-agent/data/qwen-coder-7B-repair"
    workflow = CodeFixWorkflow(
        model_path="/data1/model/qwen/Qwen/Qwen2.5-Coder-7B",
        output_dir=OUTPUT_DIR,
        max_iterations=3,
        docker_image="python:3.10-slim",
        test_timeout=3  # 每个测试用例3秒超时
    )

    total = 0
    origin_pass = 0
    fixed = 0
    not_fixed = 0

    try:
        for mbpp_data_i, metadata_i, test_result_i in tqdm(zip(mbpp_data, gen_metadata, test_results), desc="修复文件"):
            total += 1
            task_id = mbpp_data_i["task_id"]
            prompt = mbpp_data_i["text"]
            test_list = mbpp_data_i["test_list"]
            code = metadata_i["generated_code"]
            passed = test_result_i["passed"]
            if passed:
                origin_pass += 1
            else:
                fix_result = workflow.run_workflow(code, test_list, task_id, prompt)
                fix_result_path = os.path.join(OUTPUT_DIR, f"task_{task_id}_fix_result.json")
                with open(fix_result_path, 'w', encoding='utf-8') as f:
                    json.dump(fix_result, f, ensure_ascii=False, indent=4)
                if fix_result["success"]:
                    fixed += 1
                else:
                    not_fixed += 1
    finally:
        workflow.cleanup()
    
    total_fix_result = {
        "total": total,
        "origin passed": origin_pass,
        "fixed": fixed,
        "not_fixed": not_fixed
    }
    print("total fix result:", total_fix_result)
    total_result_path = os.path.join(OUTPUT_DIR, "total_result.json")
    with open(total_result_path, 'w', encoding='utf-8') as f:
        json.dump(total_fix_result, f, ensure_ascii=False)

if __name__ == "__main__":
    main()