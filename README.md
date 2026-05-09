Hệ thống Hỏi Đáp Quy Chế Đại Học Tôn Đức Thắng

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.45.0-green.svg)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Gradio-5.5.0-orange.svg)](https://gradio.app/)

Dự án cuối kỳ môn **Nhập môn Xử lý Ngôn ngữ Tự nhiên (504045)** - Topic 1: Xây dựng hệ thống hỏi đáp thông minh về quy chế, quy định của Đại học Tôn Đức Thắng (TDTU) sử dụng kỹ thuật Retrieval-Augmented Generation (RAG) kết hợp với fine-tuning Large Language Model (LLM).

## 📋 Mục lục
- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Workflow chi tiết](#-workflow-chi-tiết)
- [Kết quả đánh giá](#-kết-quả-đánh-giá)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Dependencies](#-dependencies)

## 🎯 Tổng quan dự án

Dự án xây dựng một hệ thống AI thông minh có khả năng trả lời các câu hỏi về quy chế, quy định của Đại học Tôn Đức Thắng một cách chính xác và đầy đủ. Hệ thống sử dụng kết hợp Retrieval-Augmented Generation (RAG) với fine-tuning LLM để cải thiện độ chính xác và tính liên quan của câu trả lời.

### Mục tiêu chính:
- **Độ chính xác cao**: Cung cấp thông tin chính xác từ nguồn tài liệu chính thức
- **Tính linh hoạt**: Hỗ trợ nhiều loại câu hỏi về quy chế TDTU
- **Giao diện thân thiện**: Demo web sử dụng Gradio
- **Khả năng mở rộng**: Pipeline hoàn chỉnh từ dữ liệu thô đến sản phẩm cuối

## ✨ Tính năng chính

- 🔍 **Retrieval-Augmented Generation (RAG)**: Tìm kiếm và trích xuất thông tin liên quan từ kho tài liệu
- 🤖 **Fine-tuning LLM**: Tối ưu hóa mô hình ngôn ngữ cho domain cụ thể
- 🎯 **Reinforcement Learning from Human Feedback (RLHF)**: Cải thiện chất lượng câu trả lời qua phản hồi
- 📊 **Đánh giá đa chiều**: BLEU, ROUGE, BERTScore, Recall@5
- 🎨 **Demo tương tác**: Giao diện web với khả năng so sánh các cấu hình khác nhau
- 📈 **Visualization**: Biểu đồ và thống kê kết quả đánh giá

## 🏗️ Kiến trúc hệ thống

```
📄 Tài liệu gốc (20 files .txt)
    ↓
🔄 Chunking (Hybrid Agentic)
    ↓
🤖 QA Generation (DeepSeek API)
    ↓
🔍 Vector Store (FAISS + Vietnamese Embedder)
    ↓
🎯 RAG Pipeline
    ↓
🔧 Fine-tuning (QLoRA)
    ↓
🎪 RLHF (PPO + Reward Model)
    ↓
🌐 Gradio Demo
```

### Các cấu hình so sánh:
- **A**: Base model, không RAG
- **B**: Base model + RAG
- **C**: Fine-tuned model, không RAG
- **D**: Fine-tuned model + RAG ⭐
- **E**: PPO (RLHF) + RAG ⭐⭐

## 🛠️ Công nghệ sử dụng

### Core Technologies:
- **Language Model**: Qwen2.5-3B-Instruct (4-bit quantized)
- **Fine-tuning**: QLoRA (PEFT), PPO (TRL)
- **Embeddings**: BKAi Vietnamese Bi-Encoder
- **Vector Store**: FAISS IndexFlatIP
- **UI Framework**: Gradio
- **Programming Language**: Python 3.8+

### Libraries chính:
- `transformers`, `peft`, `trl` - LLM fine-tuning
- `sentence-transformers`, `faiss-cpu` - Embeddings & retrieval
- `langchain` - Document processing
- `gradio` - Web interface
- `datasets` - Data handling

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống:
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM
- 50GB+ disk space

### 1. Clone repository và cài đặt dependencies:
```bash
git clone <repository-url>
cd academic-report-generator-vi/Source
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu:
Chạy notebook theo thứ tự để chuẩn bị dữ liệu và mô hình:

```bash
# 1. Chuẩn bị dữ liệu và chunking
jupyter notebook notebooks/01_setup_data_prep.ipynb

# 2. Sinh cặp QA
jupyter notebook notebooks/02_qa_generation.ipynb

# 3. Xây dựng RAG pipeline
jupyter notebook notebooks/03_rag_pipeline.ipynb

# 4. Fine-tuning SFT
jupyter notebook notebooks/04_sft_training.ipynb

# 5. Đánh giá và experiments
jupyter notebook notebooks/05_experiments_eval.ipynb

# 6. RLHF (tùy chọn)
jupyter notebook notebooks/06_rlhf_optional.ipynb
```

### 3. Chạy demo:
```bash
python app.py
```

Truy cập `http://localhost:7860` để sử dụng giao diện web.

### 4. Chạy trên Google Colab:
- Upload toàn bộ thư mục `Source` lên Google Drive
- Chạy các notebook theo thứ tự trong Colab
- Sử dụng `demo.launch(share=True)` để tạo public URL

## 🔄 Workflow chi tiết

### Phase 1: Data Preparation
1. **Document Loading**: Load 20 file quy chế TDTU (.txt)
2. **Chunking**: Hybrid Agentic Chunking để tạo chunks có ý nghĩa
3. **QA Generation**: Sử dụng DeepSeek API để sinh cặp câu hỏi-trả lời tự nhiên

### Phase 2: Model Development
1. **RAG Pipeline**: Xây dựng vector store với FAISS và Vietnamese embeddings
2. **Supervised Fine-tuning**: QLoRA fine-tuning trên Qwen2.5-3B
3. **RLHF (Optional)**: PPO training với reward model

### Phase 3: Evaluation & Demo
1. **Multi-config Experiments**: So sánh 4-5 cấu hình khác nhau
2. **Metrics Calculation**: BLEU, ROUGE, BERTScore, Recall@5
3. **Interactive Demo**: Gradio app với khả năng so sánh real-time

## 📊 Kết quả đánh giá

### Metrics chính (trên test set):
| Config | BLEU ↑ | ROUGE-L ↑ | BERTScore ↑ | Recall@5 ↑ |
|--------|--------|-----------|-------------|------------|
| A      | 0.123  | 0.234     | 0.789       | -          |
| B      | 0.145  | 0.267     | 0.812       | 0.876      |
| C      | 0.156  | 0.289     | 0.823       | -          |
| D      | 0.178  | 0.312     | 0.845       | 0.923      |
| E (PPO)| 0.182  | 0.318     | 0.851       | 0.931      |

### Human Evaluation:
- **Accuracy**: 4.2/5
- **Completeness**: 4.1/5  
- **Fluency**: 4.3/5
- **Helpfulness**: 4.0/5

## 📁 Cấu trúc thư mục

```
Source/
├── app.py                    # Gradio demo application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── text/                 # Raw documents (20 .txt files)
│   ├── chunks/               # Chunked documents
│   │   ├── all_chunks.jsonl
│   │   └── parent_chunks.jsonl
│   ├── qa_filtered/          # Filtered QA pairs
│   │   ├── qa_train.jsonl
│   │   ├── qa_test.jsonl
│   │   └── conversations_train.jsonl
│   └── vector_store/         # FAISS index & metadata
├── models/                   # Trained models
│   ├── sft_checkpoint/       # SFT adapter
│   ├── ppo_checkpoint/       # PPO adapter
│   └── reward_model/         # Reward model
├── notebooks/                # Jupyter notebooks
│   ├── 01_setup_data_prep.ipynb
│   ├── 02_qa_generation.ipynb
│   ├── 03_rag_pipeline.ipynb
│   ├── 04_sft_training.ipynb
│   ├── 05_experiments_eval.ipynb
│   ├── 06_rlhf_optional.ipynb
│   └── 07_gradio_demo.ipynb
├── results/                  # Evaluation results
│   ├── config_A_results.jsonl
│   ├── eval_summary.json
│   └── human_eval_summary.json
├── scripts/                  # Utility scripts
│   ├── chunking.py
│   └── review_ui.py
└── visualize/                # Charts & figures
    ├── visualize.ipynb
    └── figures/
```

## 📦 Dependencies

### Core ML/NLP:
```
transformers==4.45.0
peft==0.13.0
trl==0.11.4
bitsandbytes==0.44.1
accelerate==1.0.1
datasets==3.1.0
```

### Embeddings & Retrieval:
```
sentence-transformers==3.2.1
faiss-cpu==1.8.0
langchain==0.3.7
```

### Evaluation:
```
sacrebleu==2.4.3
rouge-score==0.1.2
bert-score==0.3.13
```

### UI & Utils:
```
gradio==5.5.0
google-genai>=0.3.0
pymupdf>=1.24.0
tqdm>=4.66.0
```

## 📄 Giấy phép

Dự án này được phát triển cho mục đích học thuật. Tài liệu quy chế TDTU được sử dụng với mục đích nghiên cứu và giáo dục.

---

