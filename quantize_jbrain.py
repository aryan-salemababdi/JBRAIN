import os
import torch
from train_sft import JBR_FinalLanguageModel

def main():
    print("شروع فاز مهندسی استنتاج (Inference Engineering)...")
    
    model_path = "Models/JBR1/checkpoints_fitness/jbrain_fitness_best_v3.pt"
    quantized_path = "Models/JBR1/checkpoints_fitness/jbrain_v3_quantized_int4.pt"

    if not os.path.exists(model_path):
        print("❌ خطا: فایل مدل v3 پیدا نشد.")
        return

    print("در حال بارگذاری مدل اصلی (FP32)...")
    model = JBR_FinalLanguageModel(vocab_size=32768, d_model=768, n_heads=12, n_layers=12)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    orig_size = os.path.getsize(model_path) / (1024 ** 2)
    print(f"📦 حجم مدل اولیه: {orig_size:.2f} مگابایت")

    print("\nدر حال فشرده‌سازی لایه‌های عصبی به INT8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint4
    )

    print("💾 در حال ذخیره مدل فشرده‌سازی شده...")
    torch.save(quantized_model.state_dict(), quantized_path)
    
    quant_size = os.path.getsize(quantized_path) / (1024 ** 2)
    print("\nکوانتیزاسیون با موفقیت انجام شد!")
    print(f"حجم مدل جدید: {quant_size:.2f} مگابایت")
    print(f"میزان کاهش حجم: {((orig_size - quant_size) / orig_size) * 100:.1f}%\n")

if __name__ == "__main__":
    main()
