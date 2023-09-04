#質問用ソースコード
import os
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, load_dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_int8_training,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.trainer_callback import TrainerCallback

OUTPUT = "./model_output"

MICRO_BATCH_SIZE = 32
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 100
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 20
VAL_SET_SIZE_HOMU = 20
TARGET_MODULES = [
    "query_key_value"
]
MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
TRAIN_DATA_PATH = "./training_data.jsonl" #さくさくむらさん形式のデータセット
device_map = "auto"
world_size = int(os.environ.get('WORLD_SIZE', 1))
ddp = world_size != 1
if ddp:
    device_map = {'':int(os.environ.get('LOCAL_RANK') or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    load_in_8bit=True 
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=LORA_R, 
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
data_prof = load_dataset("json", data_files=TRAIN_DATA_PATH)#学習用データセットの指定

train_val = data_prof["train"].train_test_split(
    test_size=VAL_SET_SIZE_HOMU, shuffle=True, seed=42 
)
train_data = train_val["train"]
val_data = train_val["test"]
def generate_prompt(data_point):#データセット入力用プロンプトの生成 (ここをどのような形式にしているのでしょうか？)
        
    if False:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:  # speaker = メジロマックイーン以外？
{data_point["dialogue"][0]["text"]}

### 回答: # speaker = メジロマックイーン？
{data_point["dialogue"][0]["text"]}"""
    else:
        result = f"""ユーザ: {data_point["instruction"]}
システム: {data_point["dialogue"][0]["text"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))


trainer = transformers.Trainer(#ここにignore_indexを指定するのでしょうか？
    model=model,
    train_dataset=train_data, #データセットの指定
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        output_dir=OUTPUT,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=None,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        optim="paged_adamw_8bit",
        auto_find_batch_size=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
model.config.use_cache = False  
trainer.train()
model.config.use_cache = True

model.save_pretrained(f"{OUTPUT}last")
