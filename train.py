import json
import pandas as pd
from typing import List, Dict
import copy
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import wandb
import datetime
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import (LoraConfig, TaskType, get_peft_model,get_peft_model_state_dict, prepare_model_for_int8_training,prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from noritication_line import noritication_line
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 基本パラメータ
def load_dataset(file_path):
    with open(file_path, 'r',encoding='utf-8-sig') as f:
        data = f.read()
        data = data.replace("}\n{", "}\\{").replace("\n", "").replace("}\\{", "}\n{")

    data_array = []

    for i, line in enumerate(data.split('\n'), start=1):
        try:
            data_item = json.loads(line)
            data_array.append(data_item)
        except json.JSONDecodeError as e:
            print(f'Error parsing JSON on line {i}: {e}')
            
            

    df = pd.DataFrame(data_array)

    return Dataset.from_pandas(df, split='train')

def transform_dialogues(json_data: List[Dict]):
    results = []
    result_id = 0

    for dialogue_data in json_data:
        dialogue_text = []
        init_inst_text = dialogue_data['instruction']
        dialogues = dialogue_data['dialogue']
        input_text =f"{dialogues[-2]['speaker']}: {dialogues[-2]['text']}\n" if len(dialogues) > 1 else ""

        for dialogue in dialogues[:-2]:
            dialogue_text.append(f"{dialogue['speaker']}: {dialogue['text']}")

        dialogue_text.append("")

        result = {
            "id": result_id,
            "instruction": (init_inst_text),
            "input": "\n".join(dialogue_text) + f"{input_text}{dialogues[-1]['speaker']}: ",
            "output": dialogues[-1]['text']
        }
        results.append(result)
        result_id += 1

    return results
model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lorappo-rinna-3.6b"
output_dir = "lorappo-rinna-3.6b-results"

def tokenize_dataset(data_point, tokenizer, ignore_index=-100):
    features = []

    for data in data_point:
      instruction_text = ""
      if data['instruction'] != "":
          instruction_text = data['instruction'] + "\n"
      prompt_full = f"[INST]\n{instruction_text}[/INST]\n{data['input']}{data['output']}{tokenizer.eos_token}"
      prompt_no_output = f"[INST]\n{instruction_text}[/INST]\n{data['input']}"

      #ignore indexに指定するほう
      if len(tokenizer.encode(prompt_full)) >= 2048:
          continue
      tokenized_full = tokenizer(
          prompt_full,
          padding='longest',
          truncation=True,
          max_length=2048,
          return_tensors='pt'
      )
      tokenized_no_output = tokenizer(
          prompt_no_output,
          padding='longest',
          truncation=True,
          max_length=2048,
          return_length=True,
          return_tensors='pt'
      )
      input_ids = tokenized_full['input_ids'][0]
      labels = copy.deepcopy(input_ids)
      source_len = tokenized_no_output['length'][0]

      labels[:source_len] = ignore_index
      
      features.append({
          'input_ids': input_ids,
          'labels': labels
      })
    return features
from transformers import AutoTokenizer

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

CUTOFF_LEN = 256  # コンテキスト長

# トークナイズ関数
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


class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )
        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
file_path = "./datasets/sliced_training_data_master.jsonl"
dataset = load_dataset(file_path)
model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
config = AutoConfig.from_pretrained(model_name,use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    load_in_8bit=True
)
new_dataset = transform_dialogues(dataset)
VAL_SET_SIZE = int(len(new_dataset) * 0.05)

new_dataset = Dataset.from_dict({k: [dic[k] for dic in new_dataset] for k in new_dataset[0]})
print(f"データセットの件数 = {len(new_dataset)}")
# 学習データと検証データの準備
train_val = new_dataset.train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=1990
)
train_data = train_val["train"]
val_data = train_val["test"]
tokenized_train = tokenize_dataset(train_data, tokenizer)
tokenized_val = tokenize_dataset(val_data, tokenizer)
collator = InstructCollator(tokenizer)
loader = DataLoader(tokenized_train, collate_fn=collator, batch_size=8, shuffle=True)
batch = next(iter(loader))
batch

#学習パラメータの指定
eval_steps = 11
save_steps = 33
logging_steps = 3
MICRO_BATCH_SIZE = 2
BATCH_SIZE = 32
from transformers import AutoModelForCausalLM

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# LoRAのパラメータ
lora_config = LoraConfig(
    r= 8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# モデルの前処理
model = prepare_model_for_int8_training(model)
# LoRAモデルの準備
model = get_peft_model(model, lora_config)
now = datetime.datetime.now()
learn_start_time = now.strftime('%Y_%m_%d_')
# 学習可能パラメータの確認
model.print_trainable_parameters()
wandb.init(project=f"macLLM_{learn_start_time}", name="display name")
# トレーナーの準備
trainer = transformers.Trainer(
    #model=model.to(torch.bfloat16),
    model = model,
    data_collator=collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    args=transformers.TrainingArguments(
        num_train_epochs=10,
        learning_rate=3e-5,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        dataloader_num_workers=12,
        logging_steps=logging_steps,
        output_dir=f"{output_dir}/{learn_start_time}",
        report_to="wandb",
        save_total_limit=1,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        auto_find_batch_size=True
    )
)

import requests
print("train start")
try:
# 学習の実行
    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True
    noritication_line("ファインチューニングが終了しました")
    # LoRAモデルの保存
    trainer.model.save_pretrained(peft_name)
    
except:
    trainer.model.save_pretrained(peft_name)
    noritication_line("ファインチューニングが異常終了しました")