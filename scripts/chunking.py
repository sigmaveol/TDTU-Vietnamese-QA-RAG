"""
Hybrid Agentic Chunking Pipeline cho tài liệu quy chế TDTU
Chunking dựa trên dấu xuống dòng \n\n (đoạn văn)
"""

import re
import json
import time
import random
import os
from typing import Optional, Set, List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
LABEL_BATCH_SIZE = 4      # Số chunk xử lý trong 1 lần gọi Gemini
RATE_LIMIT_SLEEP = 4.2    # Giây chờ giữa các API call
CHUNK_MIN        = 80     # Chars — chunk ngắn hơn sẽ bị merge vào chunk kế tiếp
CHUNK_MAX        = 800    # Chars — chunk dài hơn sẽ bị tách cơ học (fit bkai 512-token limit)


# ─────────────────────────────────────────────────────────────────────────────
# Bước 1: Chuẩn bị văn bản
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Làm sạch văn bản: chuẩn hóa xuống dòng, xóa khoảng trắng thừa.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Giữ tối đa 1 dòng trống
    result = []
    blank_count = 0
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    
    return '\n'.join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Bước 2: Tách văn bản theo \n\n (đoạn văn)
# ─────────────────────────────────────────────────────────────────────────────
def split_text(text: str) -> List[str]:
    """
    Tách văn bản thành các chunk dựa trên dấu xuống dòng \n\n.
    Mỗi đoạn văn (paragraph) là một chunk riêng.
    """
    # Làm sạch text
    text = clean_text(text)
    
    # Tách theo 2 dấu xuống dòng trở lên
    chunks = re.split(r'\n\s*\n+', text)
    
    # Loại bỏ khoảng trắng thừa và dòng trống
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    print(f"  Chunking: {len(chunks)} chunks (by paragraph breaks)")
    
    return chunks


# Bước 2b: Post-processing — Merge nhỏ + Tách lớn (Mechanical, không dùng LLM)
def merge_short_chunks(chunks: List[str], min_chars: int = CHUNK_MIN) -> List[str]:
    """
    Ghép chunk quá ngắn (<min_chars) vào chunk kế tiếp.
    Giải quyết vấn đề tiêu đề section bị tách rời thành chunk riêng 30-70 chars.
    """
    result = []
    carry = ''
    for chunk in chunks:
        combined = (carry + '\n' + chunk).strip() if carry else chunk
        if len(chunk) < min_chars:
            carry = combined  # chưa đủ dài → tiếp tục gom
        else:
            result.append(combined)
            carry = ''
    if carry:
        result.append(carry)  # flush phần còn lại
    return result


def split_large_chunk(text: str, max_chars: int = CHUNK_MAX) -> List[str]:
    """
    Tách chunk >max_chars theo cấu trúc hiện có của văn bản (không dùng LLM).

    Thứ tự ưu tiên:
    1. Bảng pipe  — dòng chứa | → header + 5 rows/chunk
    2. Bullet     — dòng bắt đầu bằng "- " hoặc "– "
    3. Numbered   — dòng bắt đầu bằng "1. " "2. " v.v.
    4. Sentence   — tách tại ". " (fallback cuối cùng)
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.split('\n')

    # 1. Bảng pipe
    pipe_lines = [l for l in lines if '|' in l and l.strip()]
    if len(pipe_lines) >= 3:
        header = pipe_lines[0]
        data   = [l for l in pipe_lines[1:]
                  if not re.match(r'^\|[\-\s|]+\|$', l)]  # bỏ dòng |---|
        prefix = '\n'.join(l for l in lines if '|' not in l).strip()
        result = []
        for i in range(0, max(len(data), 1), 5):
            group = '\n'.join([header] + data[i:i + 5])
            result.append((prefix + '\n' + group).strip() if prefix else group)
        return result

    # 2. Bullet / 3. Numbered 
    split_re = re.compile(r'(?=\n[-–•]\s|\n\d+\.\s)')
    parts = [p.strip() for p in split_re.split(text) if p.strip()]
    if len(parts) > 1:
        return merge_short_chunks(parts, min_chars=50)  # đừng tạo fragment quá nhỏ

    # 4. Sentence split (fallback) 
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks_out: List[str] = []
    current = ''
    for sent in sentences:
        if current and len(current) + len(sent) > max_chars:
            chunks_out.append(current.strip())
            current = sent
        else:
            current = (current + ' ' + sent).strip() if current else sent
    if current:
        chunks_out.append(current.strip())
    return chunks_out if chunks_out else [text]


# Bước 3: Ghi nhãn Chunk (LLM Metadata Labeling)
_LABEL_PROMPT = """Bạn là chuyên gia phân tích văn bản hành chính đại học.
Nhiệm vụ: Nhận {n} đoạn văn bản quy chế và tạo nhãn định danh cho từng đoạn theo đúng thứ tự.

Yêu cầu kỹ thuật:
1. "title": Tiêu đề (3-15 từ), phản ánh đúng tên Điều hoặc nội dung cốt lõi.
2. "summary": Tóm tắt 2-3 câu, bắt buộc giữ lại thực thể: số tín chỉ, mức phạt, thời hạn, đối tượng.
3. Không thêm thông tin ngoài văn bản, giữ nguyên thuật ngữ pháp lý.
4. Trả về đúng {n} phần tử trong JSON.

Định dạng trả về (CHỈ JSON):
{{
  "labels": [
    {{"title": "...", "summary": "..."}},
    ...
  ]
}}

DANH SÁCH {n} ĐOẠN VĂN BẢN:
{chunks_with_index}"""


def _format_chunks_for_prompt(chunks: List[str]) -> str:
    parts = [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    return '\n\n'.join(parts)


def _parse_labels(raw: str, n: int) -> List[Dict]:
    """Parse JSON response và đảm bảo đủ n labels."""
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
    data = json.loads(raw.strip())
    labels = data.get('labels', [])
    while len(labels) < n:
        labels.append({'title': '', 'summary': ''})
    return [
        {'title': str(lb.get('title', '')).strip(),
         'summary': str(lb.get('summary', '')).strip()}
        for lb in labels[:n]
    ]


def _fallback_labels(chunks: List[str]) -> List[Dict]:
    """Fallback: sinh title từ 6 từ đầu, summary từ 150 ký tự đầu."""
    return [
        {'title': ' '.join(c.split()[:6]),
         'summary': c[:150].replace('\n', ' ')}
        for c in chunks
    ]


# Gemini labeler
def label_chunks_batch(
    chunks: List[str],
    client,
    model_id: str,
    max_retries: int = 2,
) -> List[Dict]:
    """Ghi nhãn batch chunks bằng Gemini."""
    from google.genai import types

    prompt = _LABEL_PROMPT.format(
        n=len(chunks),
        chunks_with_index=_format_chunks_for_prompt(chunks),
    )
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=[types.Content(
                    role='user', parts=[types.Part(text=prompt)]
                )],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            return _parse_labels(resp.text, len(chunks))
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f'    ⚠ Gemini attempt {attempt+1} failed ({e}), retry {wait:.1f}s')
            time.sleep(wait)
    return _fallback_labels(chunks)


# DeepSeek labeler
def label_chunks_batch_deepseek(
    chunks: List[str],
    client,
    model_id: str = 'deepseek-v4-flash',
    max_retries: int = 2,
) -> List[Dict]:
    """Ghi nhãn batch chunks bằng DeepSeek."""
    prompt = _LABEL_PROMPT.format(
        n=len(chunks),
        chunks_with_index=_format_chunks_for_prompt(chunks),
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'},
                temperature=0.2,
                max_tokens=2048,
            )
            return _parse_labels(resp.choices[0].message.content, len(chunks))
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f'    ⚠ DeepSeek attempt {attempt+1} failed ({e}), retry {wait:.1f}s')
            time.sleep(wait)
    return _fallback_labels(chunks)


# Pipeline chính: HybridChunker
class HybridChunker:
    """
    Hybrid Agentic Chunker - PHIÊN BẢN ĐƠN GIẢN NHẤT.
    Chunking dựa trên dấu xuống dòng \n\n.
    """

    def __init__(
        self,
        client,
        model_id: str = 'gemini-2.0-flash',
        provider: str = 'gemini',
    ):
        self.client = client
        self.model_id = model_id
        self.provider = provider.lower()
        if self.provider not in ('gemini', 'deepseek'):
            raise ValueError(f"provider phải là 'gemini' hoặc 'deepseek'")

    def _label_batch(self, batch_texts: List[str]) -> List[Dict]:
        if self.provider == 'deepseek':
            return label_chunks_batch_deepseek(batch_texts, self.client, self.model_id)
        return label_chunks_batch(batch_texts, self.client, self.model_id)

    def process_file(
        self,
        filepath: str,
        source_file: str,
        resume_ids: Optional[Set[str]] = None,
        done_parent_ids: Optional[Set[str]] = None,
    ) -> tuple:
        """
        Xử lý một file văn bản.

        Returns:
            (parent_records, child_records)
            - parent_records : [{parent_chunk_id, source_file, text}]
            - child_records  : [{chunk_id, parent_chunk_id, source_file,
                                 text, embed_text, title, summary, ...}]
              child_records chỉ lưu parent_chunk_id (không lưu parent_text)
              → tra cứu parent text qua parent_chunks.jsonl khi cần
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        text = clean_text(raw_text)

        # Bước 2: \n\n split → parent chunks (nguyên vẹn ngữ nghĩa)
        # split_text gọi clean_text nội bộ nhưng text đã sạch → harmless double call
        chunks_raw = re.split(r'\n\s*\n+', text)
        chunks_raw = [c.strip() for c in chunks_raw if c.strip()]
        parents = merge_short_chunks(chunks_raw, min_chars=CHUNK_MIN)
        print(f"  Chunking: {len(parents)} parents (by paragraph breaks)")

        safe_name = source_file.replace(' ', '_').replace('.txt', '')[:30]

        # Gán parent_chunk_id
        parent_records = []
        paired: List[tuple] = []  # (child_text, parent_chunk_id)

        for j, parent_text in enumerate(parents):
            parent_chunk_id = f'{safe_name}_parent_{j:04d}'
            # Chỉ thêm vào parent_records nếu chưa được lưu (resume support)
            if done_parent_ids is None or parent_chunk_id not in done_parent_ids:
                parent_records.append({
                    'parent_chunk_id': parent_chunk_id,
                    'source_file'    : source_file,
                    'text'           : parent_text,
                })
            # Bước 2b: chia parent thành children nhỏ để embedding
            children = split_large_chunk(parent_text, max_chars=CHUNK_MAX)
            for child in children:
                paired.append((child, parent_chunk_id))

        print(f'  Parents: {len(parents)} | Children: {len(paired)}')

        # Lọc children đã label (resume)
        chunks_to_label = []
        skipped = 0
        for i, (child_text, parent_chunk_id) in enumerate(paired):
            chunk_id = f'{safe_name}_chunk_{i:04d}'
            if resume_ids and chunk_id in resume_ids:
                skipped += 1
                continue
            chunks_to_label.append((i, chunk_id, child_text, parent_chunk_id))

        if skipped:
            print(f'  ↩ Skipping {skipped} already-labeled chunks')

        # Bước 3: Agentic labeling theo batch
        child_records = []
        batches = [
            chunks_to_label[i:i + LABEL_BATCH_SIZE]
            for i in range(0, len(chunks_to_label), LABEL_BATCH_SIZE)
        ]

        for batch in batches:
            batch_texts = [ct for _, _, ct, _ in batch]
            labels = self._label_batch(batch_texts)
            time.sleep(RATE_LIMIT_SLEEP)

            for (idx, chunk_id, child_text, parent_chunk_id), label in zip(batch, labels):
                embed_text = f"{label['title']}\n{label['summary']}\n{child_text}".strip()
                child_records.append({
                    'chunk_id'       : chunk_id,
                    'parent_chunk_id': parent_chunk_id,   # ← tham chiếu, không lưu text
                    'source_file'    : source_file,
                    'text'           : child_text,
                    'title'          : label['title'],
                    'summary'        : label['summary'],
                    'embed_text'     : embed_text,
                    'chunk_index'    : idx,
                    'provider'       : self.provider,
                })

        return parent_records, child_records

    def process_all(
        self,
        text_dir: str,
        txt_files: List[str],
        chunks_path: str,
        parents_path: str,
    ) -> tuple:
        """
        Xử lý toàn bộ file, lưu ra 2 file:
            parents_path : parent_chunks.jsonl  — đoạn nguyên vẹn để sinh QA + RAG
            chunks_path  : all_chunks.jsonl     — child chunks để embedding

        Returns: (all_parents, all_chunks)
        """
        # --- Resume: load ids đã có
        done_chunk_ids:  Set[str] = set()
        done_parent_ids: Set[str] = set()
        all_parents: List[Dict] = []
        all_chunks:  List[Dict] = []

        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    c = json.loads(line.strip())
                    done_chunk_ids.add(c['chunk_id'])
                    all_chunks.append(c)

        if os.path.exists(parents_path):
            with open(parents_path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = json.loads(line.strip())
                    done_parent_ids.add(p['parent_chunk_id'])
                    all_parents.append(p)

        if done_chunk_ids:
            print(f'Resume: {len(done_chunk_ids)} chunks | {len(done_parent_ids)} parents')

        total_api_calls = 0

        with open(chunks_path,  'a', encoding='utf-8') as f_chunks, \
             open(parents_path, 'a', encoding='utf-8') as f_parents:

            for fname in txt_files:
                fpath = os.path.join(text_dir, fname)
                print(f'\nProcessing: {fname}')

                try:
                    parent_records, child_records = self.process_file(
                        fpath, fname,
                        resume_ids=done_chunk_ids,
                        done_parent_ids=done_parent_ids,
                    )
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f'  ✗ Error: {e}')
                    continue

                # Lưu parents mới
                new_parents = [p for p in parent_records
                               if p['parent_chunk_id'] not in done_parent_ids]
                for p in new_parents:
                    f_parents.write(json.dumps(p, ensure_ascii=False) + '\n')
                    done_parent_ids.add(p['parent_chunk_id'])
                    all_parents.append(p)

                # Lưu child chunks mới
                new_chunks = [c for c in child_records
                              if c['chunk_id'] not in done_chunk_ids]
                for c in new_chunks:
                    f_chunks.write(json.dumps(c, ensure_ascii=False) + '\n')
                    done_chunk_ids.add(c['chunk_id'])
                    all_chunks.append(c)

                n_batches = (len(new_chunks) + LABEL_BATCH_SIZE - 1) // LABEL_BATCH_SIZE
                total_api_calls += n_batches
                print(f'  ✓ {len(new_parents)} parents | {len(new_chunks)} chunks '
                      f'| {n_batches} API calls')

        print(f'\n Done. Parents: {len(all_parents)} | Chunks: {len(all_chunks)} '
              f'| API calls: {total_api_calls}')
        return all_parents, all_chunks