import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

def main():
    print("در حال بارگذاری مدل ONNX...")
    session = ort.InferenceSession("jbrain_v3_int8_mobile.onnx", providers=['CPUExecutionProvider'])
    tokenizer = Tokenizer.from_file("Data/jbrain_persian_tokenizer.json")
    eos_id = tokenizer.token_to_id("</s>")
    
    n_layers = 12
    d_model = 768
    n_heads = 12
    head_dim = 64
    
    states = {}
    for i in range(n_layers):
        states[f'past_x_{i}'] = np.zeros((1, 1, d_model), dtype=np.float32)
        states[f'prev_hr_{i}'] = np.zeros((1, n_heads, head_dim), dtype=np.float32)
        states[f'prev_hi_{i}'] = np.zeros((1, n_heads, head_dim), dtype=np.float32)

    user_query = "نفس کم آوردم چکار کنم؟"
    prompt = f"### Human:\n[وضعیت: در حال تمرین]\nنفس کم آوردم چکار کنم؟\n\n### Assistant:\n"
    input_ids = tokenizer.encode(prompt).ids
    print(input_ids)
    
    print("\n" + "="*50)
    print(f"سوال شما: {user_query}")
    print("پاسخ مربی: ", end="", flush=True)

    for token in input_ids[:-1]:
        ort_inputs = {'input_ids': np.array([[token]], dtype=np.int64)}
        ort_inputs.update(states)
        outputs = session.run(None, ort_inputs)
        for i in range(n_layers):
            idx = 1 + (i * 3)
            states[f'past_x_{i}'] = outputs[idx]
            states[f'prev_hr_{i}'] = outputs[idx+1]
            states[f'prev_hi_{i}'] = outputs[idx+2]

    current_token = input_ids[-1]
    generated_tokens = [] 
    
    for _ in range(64):
        ort_inputs = {'input_ids': np.array([[current_token]], dtype=np.int64)}
        ort_inputs.update(states)
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        
        for i in range(n_layers):
            idx = 1 + (i * 3)
            states[f'past_x_{i}'] = outputs[idx]
            states[f'prev_hr_{i}'] = outputs[idx+1]
            states[f'prev_hi_{i}'] = outputs[idx+2]
            
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == eos_id:
            break
            
        generated_tokens.append(next_token)
        decoded_text = tokenizer.decode(generated_tokens)
        print(f"\rپاسخ مربی: {decoded_text}", end="", flush=True)
        
        current_token = next_token

    print("\n" + "="*50)

if __name__ == "__main__":
    main()
