"""
Giao diện đánh giá QA pairs — Human Verification Tool (Real-time)
Chạy: python scripts/review_ui.py
Chia sẻ: copy link public in ra console, gửi cho bạn nhóm

Real-time: Mọi thay đổi đọc từ file mỗi lần render → 3 người review cùng lúc không bị conflict.
Lock file khi ghi để tránh race condition.
"""

import json
import os
import random as _rnd
import threading
import gradio as gr
from datetime import datetime
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH   = os.path.join(BASE, 'data', 'qa_filtered', 'qa_train.jsonl')
TEST_PATH    = os.path.join(BASE, 'data', 'qa_filtered', 'qa_test.jsonl')
LOG_PATH     = os.path.join(BASE, 'data', 'qa_filtered', 'review_log.jsonl')
PARENTS_PATH = os.path.join(BASE, 'data', 'chunks', 'parent_chunks.jsonl')

_FILE_LOCK = threading.Lock()

# Load parent_chunks vào dict một lần khi khởi động
_parent_map: dict[str, str] = {}
if os.path.exists(PARENTS_PATH):
    with open(PARENTS_PATH, 'r', encoding='utf-8') as _f:
        for _line in _f:
            if _line.strip():
                _p = json.loads(_line)
                _parent_map[_p['parent_chunk_id']] = _p['text']


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
def load_all_pairs() -> list[dict]:
    pairs = []
    for path, split in [(TRAIN_PATH, 'train'), (TEST_PATH, 'test')]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        p['_split'] = split
                        pairs.append(p)
    return pairs


def save_pair(pair: dict):
    path = TRAIN_PATH if pair['_split'] == 'train' else TEST_PATH
    with _FILE_LOCK:
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p['id'] == pair['id']:
                        p['human_verified'] = pair['human_verified']
                        p['verified_note']  = pair.get('verified_note', '')
                        p['verified_at']    = pair.get('verified_at', '')
                    rows.append(p)
        with open(path, 'w', encoding='utf-8') as f:
            for p in rows:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')


def save_pair_edit(pair: dict, field: str, new_value: str, reviewer: str):
    path      = TRAIN_PATH if pair['_split'] == 'train' else TEST_PATH
    old_value = pair[field]
    now       = datetime.now().isoformat()
    with _FILE_LOCK:
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p['id'] == pair['id']:
                        p[field]       = new_value
                        p['edited_at'] = now
                        p['edited_by'] = reviewer
                    rows.append(p)
        with open(path, 'w', encoding='utf-8') as f:
            for p in rows:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        with open(LOG_PATH, 'a', encoding='utf-8') as lf:
            lf.write(json.dumps({
                'pair_id'  : pair['id'],
                'action'   : f'edited_{field}',
                'field'    : field,
                'old_value': old_value,
                'new_value': new_value,
                'reviewer' : reviewer,
                'timestamp': now,
            }, ensure_ascii=False) + '\n')


def log_action(pair_id: str, action: str, reviewer: str):
    with _FILE_LOCK:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'pair_id'  : pair_id,
                'action'   : action,
                'reviewer' : reviewer,
                'timestamp': datetime.now().isoformat(),
            }, ensure_ascii=False) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Phân công công việc
# ─────────────────────────────────────────────────────────────────────────────
def assign_pairs(name1: str, name2: str, name3: str) -> str:
    """Phân công ngẫu nhiên (seed cố định): 50% / 25% / 25%."""
    names = [name1.strip(), name2.strip(), name3.strip()]
    if not all(names):
        return "⚠️ Nhập đủ tên 3 người trước khi phân công."
    if len(set(names)) < 3:
        return "⚠️ Tên 3 người phải khác nhau."

    pairs = load_all_pairs()
    n = len(pairs)
    indices = list(range(n))
    _rnd.seed(42)
    _rnd.shuffle(indices)

    # 50% - 25% - 25% (round để không bỏ sót)
    n1 = round(n * 0.50)
    n2 = round(n * 0.25)
    n3 = n - n1 - n2

    assignment: dict[str, str] = {}
    for rank, pair_idx in enumerate(indices):
        pid = pairs[pair_idx]['id']
        if rank < n1:
            assignment[pid] = names[0]
        elif rank < n1 + n2:
            assignment[pid] = names[1]
        else:
            assignment[pid] = names[2]

    # Ghi vào cả 2 file
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            continue
        with _FILE_LOCK:
            rows = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        if p['id'] in assignment:
                            p['assigned_to'] = assignment[p['id']]
                        rows.append(p)
            with open(path, 'w', encoding='utf-8') as f:
                for p in rows:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')

    counts = Counter(assignment.values())
    pct    = {name: f"{counts[name]/n*100:.0f}%" for name in names}
    return (
        f"✅ **Đã phân công {n} cặp** (seed cố định = 42, có thể phân công lại bất cứ lúc nào):\n\n"
        f"| Người review | Số cặp | Tỉ lệ |\n"
        f"|---|---|---|\n"
        f"| **{names[0]}** | {counts[names[0]]} | {pct[names[0]]} |\n"
        f"| **{names[1]}** | {counts[names[1]]} | {pct[names[1]]} |\n"
        f"| **{names[2]}** | {counts[names[2]]} | {pct[names[2]]} |"
    )


def assignment_status() -> str:
    pairs = load_all_pairs()
    assigned = [p for p in pairs if p.get('assigned_to')]
    if not assigned:
        return "_Chưa phân công. Nhập tên 3 người và nhấn **Phân công ngẫu nhiên**._"
    counts  = Counter(p['assigned_to'] for p in assigned)
    done    = Counter(
        p['assigned_to'] for p in assigned
        if p.get('human_verified') or str(p.get('verified_note', '')).startswith('rejected')
    )
    n_total = len(pairs)
    lines   = [f"**Tổng {n_total} cặp — {len(assigned)} đã phân công:**\n"]
    for name, cnt in sorted(counts.items()):
        bar   = "█" * (done[name] * 20 // cnt) + "░" * (20 - done[name] * 20 // cnt)
        lines.append(f"- **{name}**: {done[name]}/{cnt} ({done[name]/cnt*100:.0f}%)  `{bar}`")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_chunk_text(parent_chunk_id: str, source_file: str = "") -> str:
    if not parent_chunk_id:
        return "_Cặp QA này không có parent_chunk_id._"
    text = _parent_map.get(parent_chunk_id)
    if not text:
        return f"_Không tìm thấy chunk `{parent_chunk_id}` trong parent_chunks.jsonl_"
    header = f"📄 {source_file}  |  🔗 {parent_chunk_id}\n{'─'*60}\n\n"
    return header + text


def get_pool(show_all: bool, my_only: bool = False, reviewer: str = "") -> list[dict]:
    pairs = load_all_pairs()
    if my_only and reviewer.strip():
        pairs = [p for p in pairs if p.get('assigned_to', '') == reviewer.strip()]
    if show_all:
        return pairs
    return [p for p in pairs if not p.get('human_verified') and
            not str(p.get('verified_note', '')).startswith('rejected')]


def stats_text() -> str:
    pairs    = load_all_pairs()
    total    = len(pairs)
    verified = sum(1 for p in pairs if p.get('human_verified'))
    rejected = sum(1 for p in pairs if str(p.get('verified_note', '')).startswith('rejected'))
    pending  = total - verified - rejected
    train_v  = sum(1 for p in pairs if p.get('human_verified') and p['_split'] == 'train')
    test_v   = sum(1 for p in pairs if p.get('human_verified') and p['_split'] == 'test')
    n_train  = sum(1 for p in pairs if p['_split'] == 'train')
    n_test   = sum(1 for p in pairs if p['_split'] == 'test')
    return (
        f"**Tổng:** {total}  |  "
        f"✅ Xác nhận: **{verified}**  |  "
        f"❌ Từ chối: **{rejected}**  |  "
        f"⏳ Chưa xem: **{pending}**\n\n"
        f"Train: {train_v}/{n_train} verified  |  Test: {test_v}/{n_test} verified  |  "
        f"Tiến độ: **{(verified+rejected)/total*100:.1f}%**"
    )


def render_pair(idx: int, show_all: bool, my_only: bool = False, reviewer: str = ""):
    """Render pair tại idx. Trả về 12 giá trị — actions thêm stats_md thành 13."""
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        msg = (
            "### ✅ Hoàn thành! Tất cả cặp đã được review."
            if not (my_only and reviewer.strip()) else
            f"### ✅ **{reviewer}** đã review xong phần được phân công!"
        )
        return (
            msg,
            gr.update(value="", interactive=False),
            gr.update(value="", interactive=False),
            "", "", "", 0, 0,
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            "_Không có dữ liệu._",
        )
    idx  = max(0, min(int(idx), len(pool) - 1))
    pair = pool[idx]

    if pair.get('human_verified'):
        status = "✅ Đã xác nhận"
    elif str(pair.get('verified_note', '')).startswith('rejected'):
        status = "❌ Đã từ chối"
    else:
        status = "⏳ Chưa xem xét"

    edit_badge = ""
    if pair.get('edited_at'):
        t = pair['edited_at'][:16].replace('T', ' ')
        edit_badge = f"  _(✏️ sửa lần cuối: {t} bởi {pair.get('edited_by','?')})_"

    assigned_badge = ""
    if pair.get('assigned_to'):
        assigned_badge = f"  👤 *{pair['assigned_to']}*"

    chunk_text = get_chunk_text(
        pair.get('parent_chunk_id', ''),
        pair.get('source_file', ''),
    )

    scope = f" | {len(pool)} cặp của {reviewer}" if (my_only and reviewer.strip()) else f"/{len(pool)}"
    return (
        f"**[{idx+1}{scope}]** `{pair['id']}` ({pair['_split']}) — {status}{assigned_badge}{edit_badge}",
        gr.update(value=pair['question'], interactive=False),
        gr.update(value=pair['answer'],   interactive=False),
        pair.get('source_file', ''),
        pair.get('parent_chunk_id', ''),
        pair.get('verified_note', ''),
        idx,
        len(pool),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        chunk_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Navigation actions
# ─────────────────────────────────────────────────────────────────────────────
def action_verify(idx, show_all, my_only, reviewer, note):
    if not reviewer.strip():
        return (*render_pair(idx, show_all, my_only, reviewer), stats_text())
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        return (*render_pair(0, show_all, my_only, reviewer), stats_text())
    pair = pool[int(idx)]
    pair['human_verified'] = True
    pair['verified_note']  = note.strip()
    pair['verified_at']    = datetime.now().isoformat()
    save_pair(pair)
    log_action(pair['id'], 'verified', reviewer.strip())
    new_pool = get_pool(show_all, my_only, reviewer)
    next_idx = min(int(idx), max(0, len(new_pool) - 1))
    return (*render_pair(next_idx, show_all, my_only, reviewer), stats_text())


def action_reject(idx, show_all, my_only, reviewer, note):
    if not reviewer.strip():
        return (*render_pair(idx, show_all, my_only, reviewer), stats_text())
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        return (*render_pair(0, show_all, my_only, reviewer), stats_text())
    pair = pool[int(idx)]
    pair['human_verified'] = False
    pair['verified_note']  = 'rejected' + (f': {note.strip()}' if note.strip() else '')
    pair['verified_at']    = datetime.now().isoformat()
    save_pair(pair)
    log_action(pair['id'], 'rejected', reviewer.strip())
    new_pool = get_pool(show_all, my_only, reviewer)
    next_idx = min(int(idx), max(0, len(new_pool) - 1))
    return (*render_pair(next_idx, show_all, my_only, reviewer), stats_text())


def action_skip(idx, show_all, my_only, reviewer):
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        return (*render_pair(0, show_all, my_only, reviewer), stats_text())
    next_idx = (int(idx) + 1) % len(pool)
    return (*render_pair(next_idx, show_all, my_only, reviewer), stats_text())


def action_prev(idx, show_all, my_only, reviewer):
    prev_idx = max(0, int(idx) - 1)
    return (*render_pair(prev_idx, show_all, my_only, reviewer), stats_text())


# ─────────────────────────────────────────────────────────────────────────────
# Edit actions  (3 bước: bắt đầu → xác nhận → lưu/hủy)
# ─────────────────────────────────────────────────────────────────────────────
def start_edit(field: str, idx, show_all, my_only, reviewer):
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        return None, "", "", gr.update(visible=False), ""
    pair  = pool[int(idx)]
    label = "câu hỏi" if field == "question" else "câu trả lời"
    warning = (
        f"### ⚠️ Bạn chắc chắn muốn sửa {label}?\n\n"
        f"Thao tác này sẽ được **ghi lại đầy đủ** vào `review_log.jsonl`:\n"
        f"- Nội dung cũ & mới\n- Tên người sửa\n- Thời gian chính xác"
    )
    return (
        field,
        pair['question'],
        pair['answer'],
        gr.update(visible=True),
        warning,
    )


def confirm_edit_action(edit_field):
    label = "câu hỏi" if edit_field == "question" else "câu trả lời"
    return (
        gr.update(interactive=(edit_field == 'question')),
        gr.update(interactive=(edit_field == 'answer')),
        gr.update(visible=False),
        gr.update(visible=True),
        f"✏️ Đang sửa **{label}** — nhập nội dung mới rồi nhấn 💾 Lưu sửa",
    )


def cancel_confirm_action():
    return (gr.update(visible=False), "")


def save_edit_action(edit_field, idx, show_all, my_only, reviewer, q_val, a_val,
                     orig_q, orig_a):
    if not reviewer.strip():
        return (
            gr.update(value=orig_q, interactive=False),
            gr.update(value=orig_a, interactive=False),
            gr.update(visible=False),
            "⚠️ Nhập tên người review trước khi lưu.",
        )
    pool = get_pool(show_all, my_only, reviewer)
    if not pool:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False),
            "⚠️ Không tìm thấy pair.",
        )
    pair      = pool[int(idx)]
    new_value = (q_val if edit_field == 'question' else a_val).strip()
    old_value = (orig_q if edit_field == 'question' else orig_a).strip()
    if new_value == old_value:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False),
            "ℹ️ Không có thay đổi nào — không lưu.",
        )
    save_pair_edit(pair, edit_field, new_value, reviewer.strip())
    label  = "câu hỏi" if edit_field == 'question' else "câu trả lời"
    t      = datetime.now().strftime('%H:%M:%S')
    status = (
        f"✅ Đã lưu sửa **{label}** lúc {t} — "
        f"log: `{pair['id']}` bởi **{reviewer.strip()}**"
    )
    new_q = new_value if edit_field == 'question' else orig_q
    new_a = new_value if edit_field == 'answer'   else orig_a
    return (
        gr.update(value=new_q, interactive=False),
        gr.update(value=new_a, interactive=False),
        gr.update(visible=False),
        status,
    )


def cancel_edit_action(orig_q, orig_a):
    return (
        gr.update(value=orig_q, interactive=False),
        gr.update(value=orig_a, interactive=False),
        gr.update(visible=False),
        "↩ Đã hủy sửa — nội dung được giữ nguyên.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="QA Review — TDTU") as demo:
    gr.Markdown("# 📋 Human Verification — TDTU QA Dataset")

    # ── Phân công công việc ───────────────────────────────────────────────
    with gr.Accordion("🗂️ Phân công công việc (50% / 25% / 25%)", open=False):
        gr.Markdown(
            "Nhập tên 3 người review rồi nhấn **Phân công**. "
            "Hệ thống chia ngẫu nhiên theo tỉ lệ 50 / 25 / 25 với seed cố định "
            "— có thể phân công lại bất cứ lúc nào mà không mất dữ liệu đã review.\n\n"
            "Sau khi phân công, mỗi người tick **Chỉ xem phần của tôi** để lọc đúng phần của mình."
        )
        with gr.Row():
            assign_name1 = gr.Textbox(label="👤 Người 1 — 50%", placeholder="VD: Hoàng Sinh Hùng", scale=1)
            assign_name2 = gr.Textbox(label="👤 Người 2 — 25%", placeholder="VD: Nguyễn Văn A",    scale=1)
            assign_name3 = gr.Textbox(label="👤 Người 3 — 25%", placeholder="VD: Trần Thị B",      scale=1)
        assign_btn    = gr.Button("🎲 Phân công ngẫu nhiên", variant="primary", size="sm")
        assign_status = gr.Markdown(assignment_status())

    gr.Markdown("---")

    # ── Người review + bộ lọc ────────────────────────────────────────────
    with gr.Row():
        reviewer_box = gr.Textbox(
            label="👤 Tên người review (bắt buộc)",
            placeholder="VD: Hoàng Sinh Hùng",
            scale=3,
        )
        my_only_cb  = gr.Checkbox(label="Chỉ xem phần của tôi", value=False, scale=1)
        show_all_cb = gr.Checkbox(label="Hiển thị tất cả (kể cả đã review)", value=False, scale=1)

    stats_md = gr.Markdown(stats_text())
    gr.Markdown("---")

    header_md = gr.Markdown("Đang tải...")

    # ── Câu hỏi + nút Sửa ────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        q_box      = gr.Textbox(label="❓ Câu hỏi", lines=3, interactive=False, scale=10)
        edit_q_btn = gr.Button("✏️ Sửa Q", size="sm", scale=0, min_width=80)

    # ── Câu trả lời + nút Sửa ────────────────────────────────────────────
    with gr.Row(equal_height=True):
        a_box      = gr.Textbox(label="💬 Câu trả lời", lines=7, interactive=False, scale=10)
        edit_a_btn = gr.Button("✏️ Sửa A", size="sm", scale=0, min_width=80)

    # ── Dialog xác nhận sửa (ẩn mặc định) ───────────────────────────────
    with gr.Group(visible=False) as confirm_edit_group:
        gr.Markdown("---")
        confirm_warning_md = gr.Markdown("")
        with gr.Row():
            confirm_yes_btn = gr.Button("✅ Xác nhận — tôi muốn sửa", variant="primary", size="sm", scale=2)
            confirm_no_btn  = gr.Button("❌ Hủy bỏ",                  variant="stop",    size="sm", scale=1)

    # ── Thanh hành động khi đang sửa (ẩn mặc định) ──────────────────────
    with gr.Group(visible=False) as edit_actions_group:
        edit_status_md = gr.Markdown("")
        with gr.Row():
            save_edit_btn   = gr.Button("💾 Lưu sửa", variant="primary", size="sm", scale=2)
            cancel_edit_btn = gr.Button("↩ Hủy sửa",                    size="sm", scale=1)

    # ── Chunk gốc (accordion, đóng mặc định) ─────────────────────────────
    with gr.Accordion("🔍 Xem nội dung chunk gốc (để rà soát độ chính xác)", open=False):
        chunk_text_box = gr.Textbox(label="Parent chunk", lines=12, interactive=False)

    # ── Metadata ──────────────────────────────────────────────────────────
    with gr.Row():
        src_box    = gr.Textbox(label="📄 Nguồn văn bản",   interactive=False, scale=4)
        parent_box = gr.Textbox(label="🔗 Parent chunk ID", interactive=False, scale=4)

    note_box = gr.Textbox(
        label="📝 Ghi chú (tùy chọn — lý do từ chối hoặc nhận xét chất lượng)",
        placeholder="VD: Câu trả lời thiếu điều kiện / Câu hỏi không tự nhiên...",
        lines=2,
    )

    with gr.Row():
        prev_btn   = gr.Button("⬅ Trước",    size="sm",  scale=1)
        skip_btn   = gr.Button("⏭ Bỏ qua",  size="sm",  scale=1)
        reject_btn = gr.Button("❌ Từ chối", variant="stop",    size="lg", scale=2)
        verify_btn = gr.Button("✅ Xác nhận", variant="primary", size="lg", scale=2)

    # ── States ────────────────────────────────────────────────────────────
    idx_state        = gr.State(value=0)
    total_state      = gr.State(value=0)
    edit_field_state = gr.State(value=None)
    orig_q_state     = gr.State(value="")
    orig_a_state     = gr.State(value="")

    # OUTS — 13 outputs dùng chung cho navigation actions
    OUTS = [
        header_md, q_box, a_box, src_box, parent_box, note_box,
        idx_state, total_state,
        confirm_edit_group, edit_actions_group, edit_status_md,
        chunk_text_box,
        stats_md,
    ]

    # ── Phân công ─────────────────────────────────────────────────────────
    assign_btn.click(
        fn=assign_pairs,
        inputs=[assign_name1, assign_name2, assign_name3],
        outputs=[assign_status],
    )

    # ── Nút Sửa → hiện dialog xác nhận ───────────────────────────────────
    EDIT_START_OUTS = [
        edit_field_state, orig_q_state, orig_a_state,
        confirm_edit_group, confirm_warning_md,
    ]
    edit_q_btn.click(
        fn=lambda idx, sa, mo, rv: start_edit('question', idx, sa, mo, rv),
        inputs=[idx_state, show_all_cb, my_only_cb, reviewer_box],
        outputs=EDIT_START_OUTS,
    )
    edit_a_btn.click(
        fn=lambda idx, sa, mo, rv: start_edit('answer', idx, sa, mo, rv),
        inputs=[idx_state, show_all_cb, my_only_cb, reviewer_box],
        outputs=EDIT_START_OUTS,
    )

    # ── Xác nhận → mở khóa textbox ───────────────────────────────────────
    CONFIRM_OUTS = [q_box, a_box, confirm_edit_group, edit_actions_group, edit_status_md]
    confirm_yes_btn.click(fn=confirm_edit_action, inputs=[edit_field_state], outputs=CONFIRM_OUTS)
    confirm_no_btn.click(fn=cancel_confirm_action, inputs=[], outputs=[confirm_edit_group, edit_status_md])

    # ── Lưu / hủy sửa ────────────────────────────────────────────────────
    EDIT_SAVE_OUTS = [q_box, a_box, edit_actions_group, edit_status_md]
    save_edit_btn.click(
        fn=save_edit_action,
        inputs=[edit_field_state, idx_state, show_all_cb, my_only_cb, reviewer_box,
                q_box, a_box, orig_q_state, orig_a_state],
        outputs=EDIT_SAVE_OUTS,
    )
    cancel_edit_btn.click(fn=cancel_edit_action, inputs=[orig_q_state, orig_a_state], outputs=EDIT_SAVE_OUTS)

    # ── Navigation ────────────────────────────────────────────────────────
    NAV = [idx_state, show_all_cb, my_only_cb, reviewer_box]

    verify_btn.click(action_verify, NAV + [note_box], OUTS)
    reject_btn.click(action_reject, NAV + [note_box], OUTS)
    skip_btn.click(action_skip,     NAV,               OUTS)
    prev_btn.click(action_prev,     NAV,               OUTS)

    show_all_cb.change(
        fn=lambda sa, mo, rv: (*render_pair(0, sa, mo, rv), stats_text()),
        inputs=[show_all_cb, my_only_cb, reviewer_box], outputs=OUTS,
    )
    my_only_cb.change(
        fn=lambda mo, sa, rv: (*render_pair(0, sa, mo, rv), stats_text()),
        inputs=[my_only_cb, show_all_cb, reviewer_box], outputs=OUTS,
    )
    reviewer_box.change(
        fn=lambda rv, sa, mo: (*render_pair(0, sa, mo, rv), stats_text()),
        inputs=[reviewer_box, show_all_cb, my_only_cb], outputs=OUTS,
    )

    demo.load(
        fn=lambda: (*render_pair(0, False), stats_text()),
        outputs=OUTS,
    )


if __name__ == '__main__':
    pairs = load_all_pairs()
    print(f"\n{'='*50}")
    print(f"  TDTU QA Review Tool")
    print(f"  Total pairs   : {len(pairs)}")
    print(f"  Train         : {sum(1 for p in pairs if p['_split']=='train')}")
    print(f"  Test          : {sum(1 for p in pairs if p['_split']=='test')}")
    print(f"  Parent chunks : {len(_parent_map)} loaded")
    print(f"{'='*50}\n")

    PORT = 7860

    public_url = None
    try:
        from pyngrok import ngrok, conf

        NGROK_TOKEN = os.environ.get('NGROK_TOKEN', '')
        if not NGROK_TOKEN:
            env_path = os.path.join(BASE, '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('NGROK_TOKEN='):
                            NGROK_TOKEN = line.strip().split('=', 1)[1]

        if NGROK_TOKEN:
            conf.get_default().auth_token = NGROK_TOKEN
            tunnel = ngrok.connect(PORT, 'http')
            public_url = tunnel.public_url
            print(f" Public URL (ngrok): {public_url}")
        else:
            print("  NGROK_TOKEN chưa set.")
            print("   Thêm vào Source/.env:  NGROK_TOKEN=your_token")
            print("   Lấy token miễn phí: https://dashboard.ngrok.com\n")

    except ImportError:
        print("ℹ️  pyngrok chưa cài: pip install pyngrok")

    demo.launch(
        share=False,
        server_name='0.0.0.0',
        server_port=PORT,
        show_error=True,
        theme=gr.themes.Soft(),
        quiet=True,
    )

    if public_url:
        from pyngrok import ngrok as _ng
        _ng.disconnect(public_url)
