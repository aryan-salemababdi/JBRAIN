import os
import sys
import re
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from train_sft import JBR_FinalLanguageModel
except ImportError:
    print("خطا: فایل train_sft.py پیدا نشد.")
    sys.exit(1)

def clean_persian_spacing(text):
    text = re.sub(r'\s+(ات|ام|اش|مان|تان|شان|ای|اید|ایم|اند|ها|های|ه)\b', r'\1', text)
    text = re.sub(r'\s+([!؟\.،؛:])', r'\1', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def medical_triage(message, state):
    hr_match = re.search(r'ضربان.*?(\d+)', message)
    if not hr_match:
        return None 
    
    hr = int(hr_match.group(1))
    if state == "در حال استراحت":
        if 60 <= hr <= 100:
            return "ضربان قلبت کاملاً نرماله و جای نگرانی نیست. استراحت کن."
        else:
            return "این ضربان برای حالت استراحت بالاست! چند نفس عمیق بکش و اگر بهتر نشدی حتماً به پزشک مراجعه کن."
    else: # Training Phase
        if hr < 160:
            return "عالیه! ضربانت تو منطقه چربی‌سوزیه. با همین ریتم ادامه بده."
        else:
            return "ضربانت خیلی بالاست! سریعاً تمرین رو متوقف کن، بشین و نفس عمیق بکش."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    tokenizer_path = "Data/jbrain_persian_tokenizer.json"
    checkpoint_path = "Models/JBR1/checkpoints_fitness/jbrain_fitness_best_v2.pt"

    print("در حال بارگذاری توکنایزر و مدل V2...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>")

    model = JBR_FinalLanguageModel(vocab_size=32768, d_model=768, n_heads=12, n_layers=12).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    print("\n" + "="*50)
    print("شبیه‌ساز ساعت هوشمند J-BRAIN فعال شد (Router Architecture)!")
    print("="*50 + "\n")

    current_state = "در حال استراحت" 

    while True:
        try:
            print(f"\nوضعیت سنسور فعلی: [{current_state}]")
            print("برای تغییر وضعیت تایپ کن: '1' (استراحت) یا '2' (تمرین) | 'exit' برای خروج")
            user_input = input("🏋️‍♂️ سوال شما یا کد وضعیت: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input == '1':
                current_state = "در حال استراحت"
                print("وضعیت سنسور به 'استراحت' تغییر کرد.")
                continue
            elif user_input == '2':
                current_state = "در حال تمرین"
                print("وضعیت سنسور به 'تمرین' تغییر کرد.")
                continue
            elif not user_input:
                continue

            
            triage_response = medical_triage(user_input, current_state)
            
            if triage_response:
                print(f"پاسخ مربی: {triage_response}")
                print("-" * 50)
                continue
            
            formatted_prompt = f"### Human:\n{user_input}\n\n### Assistant:\n"
            
            input_ids = tokenizer.encode(formatted_prompt).ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            generated = input_tensor
            prompt_length = len(input_ids)

            print(f"پاسخ مربی: ", end="", flush=True)

            for _ in range(128):
                with torch.no_grad():
                    logits, _ = model(generated)
                    next_token_logits = logits[:, -1, :] / 0.4  
                    
                    for token_id in set(generated[0].tolist()):
                        next_token_logits[0, token_id] /= 1.1 
                        
                    values, indices = torch.topk(next_token_logits, 50)
                    probs = torch.full_like(next_token_logits, float('-inf'))
                    probs.scatter_(1, indices, values)
                    probs = F.softmax(probs, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == eos_id:
                    break
                
                generated = torch.cat((generated, next_token), dim=1)
                
                answer_ids = generated[0].tolist()[prompt_length:]
                raw_text = tokenizer.decode(answer_ids)
                cleaned_text = clean_persian_spacing(raw_text)
                
                print(f"\rپاسخ مربی: {cleaned_text}", end="", flush=True)
                                                                                
            print("\n" + "-"*50)

        except KeyboardInterrupt:
            print("\nخروج اضطراری.")
            break

if __name__ == "__main__":
    main()
