import os
import subprocess
import tempfile
from pathlib import Path
from typing import List
from utils import load_jsonl_data, construct_file_content

DOCKER_IMAGE = "python:3.10-slim"


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
            return False

        finally:
            # 4. 清理：无论成功失败，都删除容器
            subprocess.run(["docker", "rm", "-f", "temp_worker"], capture_output=True)

    if result.returncode != 0:
        print("执行失败:", result.stderr)
        print("stdout:", result.stdout)
        return False

    if "ALL_TESTS_PASSED" in result.stdout:
        print("测试通过")
        return True
    
    print("测试失败:", result.stdout)
    return False


def evaluate(data, code_dir):

    total = 0
    right = 0

    for sample in data:
        total += 1

        task_id = sample["task_id"]

        with open(os.path.join(code_dir, f"{task_id}.py"), "r") as f:
            code = f.read()
        
        testcases = sample["test_list"]
        success = run_single_sample(code, testcases)
        if success:
            right += 1

    print("total:", total)
    print("right:", right)


def main():
    data_path = "/data0/xjh/code-agent/mbpp.jsonl"
    data = load_jsonl_data(data_path)
    code_dir = "/data0/xjh/code-agent/mbpp/qwen-7B-test-first-1"
    evaluate(data, code_dir)

if __name__ == "__main__":
    main()