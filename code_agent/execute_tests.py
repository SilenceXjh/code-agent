#!/usr/bin/env python3
"""
MBPP测试执行器
在沙箱环境中执行生成的测试文件
"""

import json
import os
import subprocess
import tempfile
import time
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from utils.mbpp_utils import load_metadata


class DockerTestRunner:
    def __init__(self, test_dir, image="python:3.10-slim"):
        # 启动一个长期运行的容器
        self.container_id = subprocess.check_output([
            'docker', 'run', '-d', '--rm',
            '--network', 'none',
            '-v', f'{test_dir}:/app:ro',
            '-w', '/app',
            image,
            'tail', '-f', '/dev/null'  # 保持容器运行
        ]).decode().strip()
        print("启动docker容器:", self.container_id)
    
    def run_test(self, test_file: str, timeout: int = 5):
        """在已有容器中执行测试"""
        start_time = time.time()

        result = subprocess.run(
            ['docker', 'exec', 
             self.container_id, 
             'timeout', f'{timeout}s',  # 在容器内启动超时控制
             'python3', test_file],
            capture_output=True, text=True, timeout=timeout+2
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0 and "ALL_TESTS_PASSED" in result.stdout:
            return True, "", elapsed_time
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            error_msg = error_msg[:500]
            return False, error_msg, elapsed_time
    
    def cleanup(self):
        """清理容器"""
        subprocess.run(['docker', 'stop', self.container_id])


def categorize_error(error_msg: str) -> str:
    """分类错误类型"""
    error_lower = error_msg.lower()
    
    if '超时' in error_lower or 'timeout' in error_lower:
        return '超时'
    elif 'assertion_failed' in error_lower or 'assertionerror' in error_lower:
        return '断言失败'
    elif 'syntaxerror' in error_lower:
        return '语法错误'
    elif 'nameerror' in error_lower:
        return '未定义变量/函数'
    elif 'typeerror' in error_lower:
        return '类型错误'
    elif 'indexerror' in error_lower or 'keyerror' in error_lower:
        return '索引/键错误'
    elif 'importerror' in error_lower or 'modulenotfounderror' in error_lower:
        return '导入错误'
    elif 'zerodivisionerror' in error_lower:
        return '除零错误'
    elif 'valueerror' in error_lower:
        return '值错误'
    elif 'attributeerror' in error_lower:
        return '属性错误'
    elif 'recursionerror' in error_lower:
        return '递归深度超限'
    else:
        return '其他错误'


def execute_all_tests(
    test_dir: str,
    metadata_path: str,
    output_path: str = "test_results.jsonl",
    summary_path: str = "test_summary.json",
    timeout: int = 5
):
    """
    执行所有测试文件
    
    Args:
        test_dir: 测试文件目录
        metadata_path: 元数据文件路径
        output_path: 结果输出路径
        summary_path: 摘要输出路径
        timeout: 超时时间（秒）
    """
    print("="*60)
    print("MBPP测试执行器")
    print("="*60)
    print()
    
    # 检查Docker是否可用
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
        print("✅ Docker可用，使用Docker沙箱模式")
    except:
        print("⚠️  Docker不可用")
        return
    
    print(f"⏱️  超时设置: {timeout}秒")
    print()
    
    # 加载元数据
    print("加载元数据...")
    metadata = load_metadata(metadata_path)
    print(f"共加载 {len(metadata)} 个任务")
    print()
    
    # 清空结果文件
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 统计信息
    results = []
    passed = 0
    failed = 0
    error_types = defaultdict(int)
    total_time = 0
    
    # 执行测试
    print("开始执行测试...")
    runner = DockerTestRunner(test_dir=test_dir)
    try:
        for meta in tqdm(metadata, desc="测试进度"):
            task_id = meta['task_id']
            test_file = meta['test_file']
            
            # 检查文件是否存在
            if not os.path.exists(test_file):
                is_passed = False
                error_msg = f"测试文件不存在: {test_file}"
                elapsed_time = 0
            else:
                # 执行测试
                file_name = os.path.basename(test_file) 
                is_passed, error_msg, elapsed_time = runner.run_test(
                    file_name, timeout
                )  
            
            # 更新统计
            if is_passed:
                passed += 1
            else:
                failed += 1
                error_type = categorize_error(error_msg)
                error_types[error_type] += 1
            
            total_time += elapsed_time
            
            # 保存结果
            result = {
                'task_id': task_id,
                'test_file': test_file,
                'passed': is_passed,
                'error': error_msg if not is_passed else "",
                'error_type': categorize_error(error_msg) if not is_passed else "",
                'execution_time': round(elapsed_time, 3),
                'prompt': meta.get('prompt', ''),
                'generated_code': meta.get('generated_code', '')
            }
            results.append(result)
            
            # 实时写入结果
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    finally:
        runner.cleanup()

    # 计算统计信息
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    avg_time = (total_time / total) if total > 0 else 0
    
    # 保存摘要
    summary = {
        'total_tasks': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': round(pass_rate, 2),
        'total_time': round(total_time, 2),
        'avg_time_per_task': round(avg_time, 3),
        'timeout': timeout,
        'error_type_distribution': dict(error_types)
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"总任务数: {total}")
    print(f"通过数: {passed} ({pass_rate:.2f}%)")
    print(f"失败数: {failed} ({100-pass_rate:.2f}%)")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均耗时: {avg_time:.3f}秒/任务")
    print()
    
    if error_types:
        print("错误类型分布:")
        print("-"*60)
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors:
            percentage = (count / failed * 100) if failed > 0 else 0
            print(f"  {error_type:20s}: {count:4d} ({percentage:.1f}%)")
        print()
    
    print(f"详细结果已保存到: {output_path}")
    print(f"测试摘要已保存到: {summary_path}")
    print("="*60)
    
    return results, summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MBPP测试执行器')
    data_dir = "/data0/xjh/code-agent/data/qwen-coder-7B-test-first"
    parser.add_argument('--test-dir', default=f'{data_dir}/generations',
                        help='测试文件目录')
    parser.add_argument('--metadata', default=f'{data_dir}/generation_metadata.jsonl',
                        help='元数据文件路径')
    parser.add_argument('--output', default=f'{data_dir}/test_results.jsonl',
                        help='结果输出路径')
    parser.add_argument('--summary', default=f'{data_dir}/test_summary.json',
                        help='摘要输出路径')
    parser.add_argument('--timeout', type=int, default=5,
                        help='超时时间（秒）')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.metadata):
        print(f"错误: 找不到元数据文件 {args.metadata}")
        print("请先运行 step1_generate_code.py 生成代码")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"错误: 找不到测试目录 {args.test_dir}")
        return
    
    # 执行测试
    execute_all_tests(
        args.test_dir,
        args.metadata,
        args.output,
        args.summary,
        args.timeout
    )


if __name__ == "__main__":
    main()