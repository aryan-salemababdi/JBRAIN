import torch
from tokenizers import Tokenizer
from train_sft import JBR_FinalLanguageModel

def run_equivalence_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Equivalence Test on: {device.upper()}")

    tokenizer = Tokenizer.from_file("Data/jbrain_persian_tokenizer.json")
    model = JBR_FinalLanguageModel(vocab_size=32768, d_model=768, n_heads=12, n_layers=12).to(device)
    
    try:
        checkpoint = torch.load("Models/JBR1/checkpoints_fitness/jbrain_fitness_best.pt", map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("وزن‌های مدل با موفقیت بارگذاری شد.")
    except Exception as e:
        print(f"وزن‌ها بارگذاری نشد، تست با وزن‌های تصادفی اجرا می‌شود. خطا: {e}")
        
    model.eval()

    prompt = "### Human:\nمن الان در حال استراحت هستم. ضربانم ۱۳۰ شده چیکار کنم؟\n\n### Assistant:\n"
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"طول پرامپت: {len(input_ids)} توکن")

    with torch.inference_mode():
        logits_chunked, state_chunked = model.encode_prompt_chunked(input_tensor, chunk_size=512)
        last_logit_chunked = logits_chunked[:, -1, :]

    with torch.inference_mode():
        state_seq = None
        for i in range(len(input_ids)):
            token_in = input_tensor[:, i:i+1]
            logits_seq, state_seq = model(token_in, past_states=state_seq)
        last_logit_seq = logits_seq[:, -1, :]

    max_diff = torch.max(torch.abs(last_logit_chunked - last_logit_seq)).item()
    
    print("\n" + "="*50)
    print(f"نتیجهٔ تست هم‌ارزی حالت (State Equivalence):")
    print(f"Max Absolute Difference: {max_diff:.8f}")
    print("="*50)

    if max_diff < 1e-4:
        print("تست پاس شد! محاسبات موازی و متوالی کاملاً هم‌ارز هستند.")
        print("نتیجه: معماری و انتقال حالت پایدار است. مشکل احتمالاً از ظرفیت یا دیتاست است.")
    else:
        print(" تست رد شد! اختلاف ریاضی فراتر از حد مجاز است.")
        print("نتیجه: یک باگ ریاضی در فاز استنتاج (Inference) یا درایور حالت وجود دارد که context را فلاش می‌کند!")

if __name__ == "__main__":
    run_equivalence_test()
