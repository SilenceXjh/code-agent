"""
MBPP数据集上Qwen2.5-Coder-1.5B模型的SFT训练脚本(v3 - 带函数签名)
训练集: 第600-974条数据 (后374条)
验证集: 第511-600条数据 (90条)

改进点:
1. 只对Response部分计算loss,Instruction部分被mask
2. 在prompt中包含函数签名,确保生成的代码与测试用例匹配
"""

import json
import torch
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List
from dataclasses import dataclass
import os


def extract_function_signature(code: str) -> str:
    """
    从代码中提取函数签名
    
    Args:
        code: Python代码字符串
    
    Returns:
        函数签名字符串,如 "def similar_elements(test_tup1, test_tup2):"
    """
    lines = code.strip().split('\n')
    for line in lines:
        # line = line.strip()
        if line.startswith('def '):
            return line
    return ""


def load_mbpp_data(file_path: str, start_idx: int, end_idx: int) -> List[Dict]:
    """
    加载MBPP数据集的指定范围
    
    Args:
        file_path: MBPP jsonl文件路径
        start_idx: 起始索引 (包含)
        end_idx: 结束索引 (不包含)
    
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    return all_data[start_idx:end_idx]


def format_prompt_with_signature(example: Dict) -> tuple:
    """
    将MBPP数据格式化为训练提示,包含函数签名
    
    返回: (instruction_text, full_text)
    instruction_text: 只包含指令和函数签名的部分(用于masking)
    full_text: 完整的训练文本
    """
    instruction = example['text']
    code = example['code']
    
    # 提取函数签名
    signature = extract_function_signature(code)
    
    if signature:
        # 如果成功提取到签名,将其加入prompt
        instruction_text = f"""### Instruction:
{instruction}

### Function Signature:
{signature}

### Response:
"""
        
        full_text = f"""### Instruction:
{instruction}

### Function Signature:
{signature}

### Response:
{code}"""
    else:
        # 如果没有提取到签名(异常情况),使用原始格式
        instruction_text = f"""### Instruction:
{instruction}

### Response:
"""
        
        full_text = f"""### Instruction:
{instruction}

### Response:
{code}"""
    
    return instruction_text, full_text


def preprocess_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    """
    预处理函数: 将文本转换为模型输入
    只对response部分计算loss,instruction和函数签名部分被mask为-100
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for text, code in zip(examples['text'], examples['code']):
        # 格式化prompt
        instruction_text, full_text = format_prompt_with_signature({
            'text': text,
            'code': code
        })
        
        # Tokenize instruction部分(包括函数签名,但不包括response内容)
        instruction_tokens = tokenizer.encode(
            instruction_text,
            add_special_tokens=True,
            truncation=False
        )
        
        # Tokenize完整文本
        full_tokens = tokenizer.encode(
            full_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        
        # 创建labels: instruction和签名部分设为-100,response部分保持原值
        labels = full_tokens.copy()
        instruction_length = len(instruction_tokens)
        
        # Mask instruction和函数签名部分
        for i in range(min(instruction_length, len(labels))):
            labels[i] = -100
        
        # 创建attention mask
        attention_mask = [1] * len(full_tokens)
        
        input_ids_list.append(full_tokens)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }


@dataclass
class DataCollatorForSupervisedDataset:
    """
    自定义data collator,处理padding并保持labels的masking
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        
        # 动态padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100  # padding位置也不计算loss
        )
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask) for mask in attention_mask],
            batch_first=True,
            padding_value=0
        )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def main():
    # ==================== 配置参数 ====================
    MBPP_FILE = "mbpp.jsonl"  # MBPP数据集路径
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"  # 模型名称
    OUTPUT_DIR = "./mbpp_sft_output"  # 输出目录
    
    # 数据划分
    TRAIN_START = 600  # 训练集起始索引
    TRAIN_END = 974    # 训练集结束索引 (374条数据)
    VAL_START = 511    # 验证集起始索引
    VAL_END = 600      # 验证集结束索引 (90条数据)
    
    MAX_LENGTH = 512   # 最大序列长度
    
    # ==================== 加载数据 ====================
    print("Loading MBPP dataset...")
    train_data = load_mbpp_data(MBPP_FILE, TRAIN_START, TRAIN_END)
    val_data = load_mbpp_data(MBPP_FILE, VAL_START, VAL_END)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # 验证函数签名提取
    print("\n" + "="*80)
    print("Verifying function signature extraction:")
    print("="*80)
    sample = train_data[0]
    print(f"Sample code:\n{sample['code']}\n")
    signature = extract_function_signature(sample['code'])
    print(f"Extracted signature: {signature}")
    instruction_text, full_text = format_prompt_with_signature(sample)
    print(f"\nFormatted prompt:\n{full_text}")
    print(f"\nFormatted instruction text:\n{instruction_text}")
    print("="*80 + "\n")
    
    # 转换为Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # ==================== 加载模型和分词器 ====================
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # ==================== 配置LoRA ====================
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA秩
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],  # Qwen2.5的attention和MLP层
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ==================== 预处理数据 ====================
    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    # 验证数据处理
    print("\n" + "="*80)
    print("Sample data verification:")
    print("="*80)
    sample_data = tokenized_train[0]
    print(f"Input IDs length: {len(sample_data['input_ids'])}")
    print(f"Labels length: {len(sample_data['labels'])}")
    print(f"Number of -100 in labels (masked tokens): {sample_data['labels'].count(-100)}")
    print(f"Number of valid labels: {len([l for l in sample_data['labels'] if l != -100])}")
    
    # 解码查看
    print("\nDecoded input:")
    print(tokenizer.decode(sample_data['input_ids']))
    print("\nMasked part (instruction + signature):")
    masked_tokens = [token_id for token_id, label in zip(sample_data['input_ids'], sample_data['labels']) if label == -100]
    print(tokenizer.decode(masked_tokens))
    print("\nNon-masked part (response - will compute loss):")
    response_tokens = [token_id for token_id, label in zip(sample_data['input_ids'], sample_data['labels']) if label != -100]
    print(tokenizer.decode(response_tokens))
    print("="*80 + "\n")
    
    # ==================== 训练参数 ====================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 有效batch size = 4 * 4 = 16
        learning_rate=2e-4,
        # warmup_steps=100,
        warmup_ratio=0.1,
        logging_steps=10,
        # save_steps=100,
        # eval_steps=100,
        # eval_strategy="steps",
        # save_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # ==================== Data Collator ====================
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # ==================== 训练器 ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # ==================== 开始训练 ====================
    print("Starting training...")
    trainer.train()
    
    # ==================== 保存模型 ====================
    print("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    
    print("Training completed!")
    print(f"Model saved to {OUTPUT_DIR}/final_model")


if __name__ == "__main__":
    main()