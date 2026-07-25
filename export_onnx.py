import os
import torch
import torch.nn as nn
from train_sft import JBR_FinalLanguageModel

class JBR_ONNX_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = len(model.layers)

    def forward(self, input_ids, *flat_past_states):
        """
        مدل‌های موبایل (ONNX) نمی‌توانند لیست دریافت کنند. 
        این تابع ۳۶ تنسور مجزا را می‌گیرد و دوباره تبدیل به لیستِ تاپل‌ها می‌کند.
        """
        past_states = []
        if len(flat_past_states) > 0:
            for i in range(self.n_layers):
                idx = i * 3
                past_states.append((
                    flat_past_states[idx],     # past_x
                    flat_past_states[idx+1],   # prev_h_r
                    flat_past_states[idx+2]    # prev_h_i
                ))
        else:
            past_states = None

        logits, next_states = self.model(input_ids, past_states)

        flat_next_states = []
        for state in next_states:
            flat_next_states.extend(list(state))

        return logits[:, -1:, :], *flat_next_states


def main():
    print("شروع عملیات Export به ONNX...")
    
    vocab_size = 32768
    d_model = 768
    n_heads = 12
    head_dim = d_model // n_heads
    n_layers = 12
    device = "cpu"

    checkpoint_path = "Models/JBR1/checkpoints_fitness/jbrain_fitness_best_v3.pt"
    onnx_fp32_path = "jbrain_v3_fp32.onnx"
    onnx_int8_path = "jbrain_v3_int8_mobile.onnx"

    print("در حال بارگذاری مدل اصلی...")
    model = JBR_FinalLanguageModel(vocab_size, d_model, n_heads, n_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    wrapped_model = JBR_ONNX_Wrapper(model).eval()

    print("در حال ساخت گراف محاسباتی (Tracing)...")
    batch_size = 1
    seq_len = 1 # for inference graph (L=1)
    
    dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dummy_past_states = []
    
    for _ in range(n_layers):
        dummy_past_states.append(torch.zeros(batch_size, 1, d_model))             # past_x
        dummy_past_states.append(torch.zeros(batch_size, n_heads, head_dim))      # prev_h_r
        dummy_past_states.append(torch.zeros(batch_size, n_heads, head_dim))      # prev_h_i

    dummy_inputs = tuple([dummy_input_ids] + dummy_past_states)

    input_names = ['input_ids']
    output_names = ['logits']
    dynamic_axes = {
        'input_ids': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }

    for i in range(n_layers):
        input_names.extend([f'past_x_{i}', f'prev_hr_{i}', f'prev_hi_{i}'])
        output_names.extend([f'next_x_{i}', f'next_hr_{i}', f'next_hi_{i}'])
        
        dynamic_axes[f'past_x_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'prev_hr_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'prev_hi_{i}'] = {0: 'batch_size'}
        
        dynamic_axes[f'next_x_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'next_hr_{i}'] = {0: 'batch_size'}
        dynamic_axes[f'next_hi_{i}'] = {0: 'batch_size'}

    print(f"📦 در حال ذخیره فایل ONNX خام در: {onnx_fp32_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_inputs,
        onnx_fp32_path,
        export_params=True,
        opset_version=14,         
        do_constant_folding=True, 
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("اکسپورت FP32 موفقیت‌آمیز بود!")

    print("\nدر حال فشرده‌سازی ONNX به فرمت INT8 برای دیوایس لبه...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"مدل نهایی موبایل با موفقیت ساخته شد: {onnx_int8_path}")
        
        fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
        int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
        print(f"حجم قبل از فشرده‌سازی: {fp32_size:.2f} MB")
        print(f"حجم آماده‌ی نصب روی ساعت: {int8_size:.2f} MB")
        
    except ImportError:
        print("کتابخانه onnxruntime نصب نیست. لطفاً دستور زیر را اجرا کنید:")
        print("pip install onnx onnxruntime")

if __name__ == "__main__":
    main()
