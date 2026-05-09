import os, torch, pickle
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

BASE       = os.path.dirname(__file__)
MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
SFT_PATH   = os.path.join(BASE, "models", "sft_checkpoint")
PPO_PATH   = os.path.join(BASE, "models", "ppo_checkpoint")
REWARD_PATH = os.path.join(BASE, "models", "reward_model")
INDEX_DIR  = os.path.join(BASE, "data", "vector_store", "faiss_index")

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI của Trường Đại học Tôn Đức Thắng (TDTU). "
    "Bạn trả lời các câu hỏi về quy chế, quy định, chính sách của trường một cách chính xác và đầy đủ. "
    "Trả lời bằng tiếng Việt. Nếu không có đủ thông tin, hãy nói rõ điều đó."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = (
    torch.bfloat16 if device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 30e9
    else torch.float16 if device == "cuda"
    else torch.float32
)
print(f"Device: {device}  |  dtype: {COMPUTE_DTYPE}")

# --- FAISS ---
print("Loading FAISS index...")
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
with open(os.path.join(INDEX_DIR, "index.pkl"), "rb") as f:
    index_chunks = pickle.load(f)

# --- Embedder ---
print("Loading embedding model...")
embedder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cpu")

# --- Tokenizer & base/sft models ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(SFT_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
base_model.eval()

print("Loading fine-tuned model (SFT adapter)...")
ft_model = PeftModel.from_pretrained(base_model, SFT_PATH, adapter_name="sft")
ft_model.eval()

# --- PPO adapter (optional) — loaded onto the same ft_model via named adapters ---
RLHF_AVAILABLE = False
if os.path.exists(PPO_PATH):
    print("Loading PPO adapter (Config E)...")
    ft_model.load_adapter(PPO_PATH, adapter_name="ppo")
    ft_model.set_adapter("sft")   # keep SFT active by default
    RLHF_AVAILABLE = True
    print("PPO adapter loaded.")
else:
    print(f"PPO checkpoint not found at {PPO_PATH}")

# --- Reward model (optional) ---
reward_model = None
reward_tokenizer = None
if os.path.exists(REWARD_PATH):
    print("Loading reward model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_PATH, trust_remote_code=True)
    _reward_base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=1,
        quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    reward_model = PeftModel.from_pretrained(_reward_base, REWARD_PATH)
    reward_model.eval()
    print("Reward model loaded.")
else:
    print(f"Reward model not found at {REWARD_PATH}")

REWARD_AVAILABLE = reward_model is not None
print("All components loaded.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> list:
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_vec, k)
    return [{"chunk": index_chunks[i], "score": float(scores[0][j])}
            for j, i in enumerate(indices[0]) if i < len(index_chunks)]


def build_prompt(question: str, context_chunks: list | None) -> str:
    if context_chunks:
        context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
        user_content = (
            f"Dựa vào các đoạn văn bản sau từ quy chế TDTU:\n\n"
            f"{context_str}\n\nCâu hỏi: {question}"
        )
    else:
        user_content = question
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(model, prompt: str, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=IM_END_ID,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def reward_score(text: str) -> float:
    if not REWARD_AVAILABLE:
        return float("nan")
    inputs = reward_tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=512).to(reward_model.device)
    with torch.no_grad():
        return reward_model(**inputs).logits.squeeze().item()


# ---------------------------------------------------------------------------
# Gradio tab functions
# ---------------------------------------------------------------------------

EXAMPLES_VI = [
    "Sinh viên bị đình chỉ học tập trong trường hợp nào?",
    "Điều kiện để được xét học bổng khuyến khích học tập là gì?",
    "Quy định về trang phục của sinh viên trong trường như thế nào?",
    "Sinh viên có thể chuyển ngành học không? Điều kiện và thủ tục ra sao?",
    "Mức xử lý kỷ luật khi sinh viên gian lận thi cử là gì?",
    "Điều kiện tốt nghiệp đại học tại TDTU là gì?",
]


def answer_question(question: str, use_rag: bool, use_finetuned: bool, top_k: int):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", ""
    model = ft_model if use_finetuned else base_model
    if use_rag:
        results = retrieve(question, k=top_k)
        top3_texts = [r["chunk"]["text"] for r in results[:3]]
        context_display = "\n\n".join(
            f"**[{i+1}]** (Score: {r['score']:.3f}) *{r['chunk']['source_file']}*\n\n"
            f"{r['chunk']['text'][:300]}{'...' if len(r['chunk']['text']) > 300 else ''}"
            for i, r in enumerate(results)
        )
    else:
        top3_texts = []
        context_display = "*(RAG không được bật)*"
    prompt = build_prompt(question, top3_texts if use_rag else None)
    return generate(model, prompt), context_display


def compare_all_configs(question: str):
    if not question.strip():
        return ("Vui lòng nhập câu hỏi.",) * 4
    results = retrieve(question, k=5)
    top3_texts = [r["chunk"]["text"] for r in results[:3]]
    ans_a = generate(base_model, build_prompt(question, None))
    ans_b = generate(base_model, build_prompt(question, top3_texts))
    ans_c = generate(ft_model,   build_prompt(question, None))
    ans_d = generate(ft_model,   build_prompt(question, top3_texts))
    return ans_a, ans_b, ans_c, ans_d


def compare_rlhf(question: str):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", "Vui lòng nhập câu hỏi.", ""
    results    = retrieve(question, k=5)
    top3_texts = [r["chunk"]["text"] for r in results[:3]]
    prompt     = build_prompt(question, top3_texts)
    ft_model.set_adapter("sft")
    ans_d   = generate(ft_model, prompt)
    score_d = reward_score(ans_d)
    if RLHF_AVAILABLE:
        ft_model.set_adapter("ppo")
        ans_e = generate(ft_model, prompt)
        ft_model.set_adapter("sft")
    else:
        ans_e = "(PPO model chưa có — chạy NB06 trước)"
    score_e = reward_score(ans_e)
    if REWARD_AVAILABLE and RLHF_AVAILABLE:
        score_info = (
            f"**Reward scores** (higher = better)\n\n"
            f"- Config D: `{score_d:.4f}`\n"
            f"- Config E (PPO): `{score_e:.4f}`\n"
            f"- Improvement: `{score_e - score_d:+.4f}`"
        )
    else:
        score_info = "*(Reward model hoặc PPO model chưa được load)*"
    return ans_d, ans_e, score_info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TDTU QA System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Hệ thống Hỏi-Đáp Quy chế TDTU\n"
        "*Nhập môn Xử lý Ngôn ngữ Tự nhiên (504045) — Topic 1*"
    )

    with gr.Tab("Hỏi đáp"):
        with gr.Row():
            with gr.Column(scale=2):
                question_box = gr.Textbox(
                    label="Câu hỏi của bạn",
                    placeholder="Ví dụ: Điều kiện để được xét học bổng khuyến khích học tập là gì?",
                    lines=3,
                )
                with gr.Row():
                    use_rag_cb  = gr.Checkbox(label="Sử dụng RAG", value=True)
                    use_ft_cb   = gr.Checkbox(label="Sử dụng mô hình fine-tuned", value=True)
                    topk_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K chunks")
                with gr.Row():
                    submit_btn = gr.Button("Gửi câu hỏi", variant="primary")
                    gr.ClearButton([question_box])
                gr.Examples(examples=EXAMPLES_VI, inputs=question_box)
            with gr.Column(scale=3):
                answer_box  = gr.Textbox(label="Câu trả lời", lines=8, interactive=False)
                context_box = gr.Markdown(label="Tài liệu tham khảo (RAG chunks)")
        submit_btn.click(
            fn=answer_question,
            inputs=[question_box, use_rag_cb, use_ft_cb, topk_slider],
            outputs=[answer_box, context_box],
        )

    with gr.Tab("So sánh 4 cấu hình (A/B/C/D)"):
        gr.Markdown(
            "**A:** Base, no RAG &nbsp;|&nbsp; **B:** Base + RAG &nbsp;|&nbsp; "
            "**C:** Fine-tuned, no RAG &nbsp;|&nbsp; **D:** Fine-tuned + RAG"
        )
        compare_q   = gr.Textbox(label="Nhập câu hỏi để so sánh", lines=2)
        compare_btn = gr.Button("So sánh tất cả", variant="secondary")
        with gr.Row():
            cfg_a_box = gr.Textbox(label="Config A (Base, no RAG)",        lines=7)
            cfg_b_box = gr.Textbox(label="Config B (Base + RAG)",           lines=7)
        with gr.Row():
            cfg_c_box = gr.Textbox(label="Config C (Fine-tuned, no RAG)", lines=7)
            cfg_d_box = gr.Textbox(label="Config D (Fine-tuned + RAG) ★", lines=7)
        compare_btn.click(
            fn=compare_all_configs,
            inputs=[compare_q],
            outputs=[cfg_a_box, cfg_b_box, cfg_c_box, cfg_d_box],
        )

    with gr.Tab("Config D vs PPO (Config E)"):
        rlhf_status = (
            "✅ PPO + Reward model sẵn sàng" if (RLHF_AVAILABLE and REWARD_AVAILABLE)
            else "⚠️ Cần chạy NB06 trước để có PPO và Reward model"
        )
        gr.Markdown(
            f"So sánh **Config D** (Fine-tuned + RAG) với **Config E** (PPO/RLHF + RAG)\n\n{rlhf_status}"
        )
        rlhf_q   = gr.Textbox(label="Câu hỏi", lines=2)
        rlhf_btn = gr.Button("So sánh D vs E", variant="secondary")
        with gr.Row():
            cfg_d_rlhf = gr.Textbox(label="D — Fine-tuned + RAG",   lines=8)
            cfg_e_rlhf = gr.Textbox(label="E — PPO (RLHF) + RAG ★", lines=8)
        score_md = gr.Markdown(label="Reward scores")
        rlhf_btn.click(
            fn=compare_rlhf,
            inputs=[rlhf_q],
            outputs=[cfg_d_rlhf, cfg_e_rlhf, score_md],
        )
        gr.Examples(examples=EXAMPLES_VI, inputs=rlhf_q)

    with gr.Tab("Thông tin mô hình"):
        ppo_status    = "✅ Available" if RLHF_AVAILABLE   else "⚠️ Not available"
        reward_status = "✅ Available" if REWARD_AVAILABLE else "⚠️ Not available"
        gr.Markdown(f"""
## Chi tiết mô hình

| Thành phần | Chi tiết |
|---|---|
| Base LLM | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Fine-tuning | QLoRA (r=16, alpha=32) via SFTTrainer |
| RLHF | PPO (2 epochs) |
| Reward model | LoRA r=8, `AutoModelForSequenceClassification` |
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` |
| Vector Store | FAISS IndexFlatIP |
| PPO model | {ppo_status} |
| Reward model | {reward_status} |
        """)

demo.launch()
