import os
from pathlib import Path
import subprocess
import tempfile

from openai import OpenAI
from utils import load_jsonl_data, ds_api_generate, extract_python_code, construct_file_content, get_testcases

data_path = "/data0/xjh/code-agent/HumanEval.jsonl"
output_path = "/data0/xjh/code-agent/human_eval/ds-with-feedback"

code_path = "/data0/xjh/code-agent/human_eval/ds-test-first"

DOCKER_IMAGE = "python:3.10-slim"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data = load_jsonl_data(data_path)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

os.makedirs(output_path, exist_ok=True)


def run_single_sample(code_str, testcases):
    with tempfile.TemporaryDirectory(dir="/data0/xjh/tmp") as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "solution.py").write_text(construct_file_content(code_str, testcases))

        create_cmd = [
            "docker", "create",
            "--name", "temp_worker",  # 指定容器名称
            "--workdir", "/app",
            DOCKER_IMAGE,
            "sh", "-c", "python solution.py"
        ]

        try:
            # 创建容器并获取 ID
            subprocess.run(create_cmd, check=True, capture_output=True)

            # 2. 拷贝文件 (docker cp)
            # 直接将整个临时目录下的内容拷贝到容器的 /app 目录
            # 注意：src 路径最后加个 / 会拷贝目录下的内容，而不是目录本身
            subprocess.run(["docker", "cp", f"{tmpdir}/.", "temp_worker:/app"], 
                           check=True,
                           capture_output=True,
                           text=True)

            # 3. 启动并等待结果 (docker start)
            # -a (attach) 会让 subprocess 等待容器运行结束并获取输出
            result = subprocess.run(
                ["docker", "start", "-a", "temp_worker"],
                capture_output=True,
                text=True,
                timeout=10
            )

        except Exception as e:
            print(e)
            return False, str(e)

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

    if result.returncode != 0:
        print("执行失败:", result.stderr)
        print("stdout:", result.stdout)
        return False, result.stdout

    if "ALL_TESTS_PASSED" in result.stdout:
        print("测试通过")
        return True, None
    
    print("测试失败:", result.stdout)
    return False, result.stdout

total = 0
right = 0
repaired = [0,0,0]

for sample in data:
    total += 1
    task_id = sample["task_id"]
    task_id = task_id.replace('/', '_')
    
    with open(os.path.join(code_path, f"{task_id}.py"), "r") as f:
        code = f.read()

    testcases = get_testcases(sample["test"], sample["entry_point"])

    success, message = run_single_sample(code, testcases)
    if success:
        right += 1
    else:
        for i in range(3):
            full_prompt = f"""Please fix a python function base on the function description, current code and test feedback.

Function Description:
{sample["prompt"]}
        
Current code:
```python
{code}
```

Test feedBack: 
{message}

Provide only the fixed Python function code without any explanation."""
            
            generated_text = ds_api_generate(full_prompt, client)
            code = extract_python_code(generated_text)
            ok, message = run_single_sample(code, testcases)
            if ok:
                repaired[i] += 1
                with open(os.path.join(output_path, f"{task_id}.py"), "w") as f:
                    f.write(code)
                break

print("total:", total)
print("right:", right)
print("repaired:", repaired)
