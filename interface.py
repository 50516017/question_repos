import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate(text, history=None):
  if history:
      history += "\n"
  else:
      history = ""
  prompt_no_output = f"[INST]\n[/INST]\n{history}トレーナー: {text}\nメジロマックイーン: "
  token_ids = tokenizer.encode(prompt_no_output, add_special_tokens=False, return_tensors="pt")

  with torch.no_grad():
      output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=2000,
        # temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
        

  return "========\n" + tokenizer.decode(output_ids.tolist()[0])

OUTPUT = "./lorappo-rinna-3.6b/"
MODELDIR = ""
LORA_WEIGHTS=f'{OUTPUT}{MODELDIR}'
MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
device_map = "auto"
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    load_in_8bit=True,   
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# LoRAモデルの準備
model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    device_map="auto"
)

model.eval()
print(generate("おはよう！"))
print(generate("こんにちわ"))
print(generate("日本の首都は？"))
print(generate("自己紹介してください"))
print(generate("おはよう"))
print(generate("あなたは誰？"))
print(generate("自己紹介してください"))
print(generate("トウカイテイオーとは？"))
print(generate("日本で一番高い山は？"))
print(generate("ゴールドシップって知ってる？"))
print(generate("あなたの尊敬する人は？"))
print(generate("今日の予定は？"))
print(generate("甘いもの食べに行こうよ。"))
print(generate("好きな食べ物は？"))
print(generate("ウマ娘で一番かわいいのは誰？"))
print(generate("お腹減ったね、マックイーン。"))
print(generate("マックイーンは何歳？"))
print(generate("自己紹介してみて！"))
print(generate("自己PRしてみて"))
print(generate("自己紹介してみて！"))
print(generate("メジロマックイーンってなんだっけ？"))