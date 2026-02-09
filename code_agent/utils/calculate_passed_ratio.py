import json
import sys
import argparse

def calculate_passed_ratio(jsonl_file_path):
    """
    计算task_id在11-510之间的条目中passed为true的比例
    """
    total_count = 0
    passed_count = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            line_number = 0
            for line in file:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    
                    # 检查task_id是否在11-510之间
                    if task_id is not None and 11 <= task_id <= 510:
                        total_count += 1
                        
                        # 检查passed是否为true（注意：JSON中的布尔值是true/false，Python中会解析为True/False）
                        if data.get('passed') is True:
                            passed_count += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_number}行无法解析JSON: {e}")
                    print(f"行内容: {line[:100]}...")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件 '{jsonl_file_path}' 未找到")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生错误: {e}")
        return None
    
    if total_count == 0:
        print("警告: 没有找到task_id在11-510之间的条目")
        return 0
    
    passed_ratio = passed_count / total_count
    
    # 打印详细统计信息
    print(f"统计结果:")
    print(f"task_id范围: 11-510")
    print(f"总条目数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {total_count - passed_count}")
    print(f"通过率: {passed_ratio:.2%} ({passed_ratio:.4f})")
    
    return passed_ratio


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中task_id在11-510之间条目的通过率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--jsonl_file',
        type=str,
        help='JSONL文件路径'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"正在分析文件: {args.jsonl_file}")
    
    # 计算通过率
    ratio = calculate_passed_ratio(args.jsonl_file)
    
    if ratio is not None:
        print(f"\n最终结果: task_id为11-510的条目中passed为true的比例是 {ratio:.2%}")


if __name__ == "__main__":
    main()