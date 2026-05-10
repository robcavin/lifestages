"""
Life Stages Video Generator
Crops/orders portrait images and generates LTX-2.3 transition clips between them.

image_list_state: list of {"id": unique_str, "path": original_path_str}
crop_data_state:  dict of str(list_index) -> cropped_file_path
"""

import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import gradio as gr
from PIL import Image

# ── Config — all overridable via environment variables ────────────────────────

# LTX model paths (only needed for video generation)
CHECKPOINT     = os.environ.get(
    "LTX_CHECKPOINT",
    str(Path.home() / ".cache/huggingface/hub/models--Lightricks--LTX-Video"
        "/snapshots/main/ltx-2.3-22b-dev.safetensors")
)
DISTILLED_LORA = os.environ.get(
    "LTX_DISTILLED_LORA",
    str(Path.home() / ".cache/huggingface/hub/models--Lightricks--LTX-Video"
        "/snapshots/main/ltx-2.3-22b-distilled-lora.safetensors")
)
# Python interpreter that has ltx_pipelines installed
LTX_PYTHON     = os.environ.get("LTX_PYTHON", sys.executable)
# Working directory for the LTX subprocess (repo root)
LTX_CWD        = os.environ.get("LTX_CWD", str(Path(__file__).parent))

APP_DIR        = Path(__file__).parent
OUTPUT_DIR     = APP_DIR / "output"
CROP_DIR       = OUTPUT_DIR / "crops"
LAST_DIR_FILE  = APP_DIR / "last_dir.txt"
SESSION_NAME   = ".lifestages_session.json"
OUTPUT_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff"}
THUMB_SIZE = (150, 200)


# ── JS ────────────────────────────────────────────────────────────────────────

SORTABLE_SETUP_JS = """
async () => {
    if (!window.Sortable) {
        await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js';
            s.onload = resolve; s.onerror = reject;
            document.head.appendChild(s);
        });
    }

    // Write to a <textarea> (used for order-box)
    function writeTextarea(elemId, value) {
        const tb = document.querySelector('#' + elemId + ' textarea');
        if (!tb) return;
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(tb, value);
        tb.dispatchEvent(new Event('input', {bubbles: true}));
    }

    // Write to a <input type="number"> (used for select-box)
    function writeNumber(elemId, value) {
        const inp = document.querySelector('#' + elemId + ' input[type="number"]');
        if (!inp) return;
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        setter.call(inp, String(value));
        inp.dispatchEvent(new Event('input',  {bubbles: true}));
        inp.dispatchEvent(new Event('change', {bubbles: true}));
    }

    function initSortable() {
        const container = document.getElementById('ls-sortable');
        if (!container || container._ls_init) return;
        container._ls_init = true;

        container.addEventListener('click', function(e) {
            const card = e.target.closest('.ls-card');
            if (!card) return;
            container.querySelectorAll('.ls-card').forEach(c => c.style.outline = 'none');
            card.style.outline = '3px solid #6366f1';
            const cards = Array.from(container.querySelectorAll('.ls-card'));
            writeNumber('ls-select-box', cards.indexOf(card));
        });

        new Sortable(container, {
            animation: 150,
            ghostClass: 'ls-ghost',
            onEnd: () => {
                const cards = container.querySelectorAll('.ls-card');
                cards.forEach((c, i) => {
                    const num = c.querySelector('.ls-num');
                    if (num) num.textContent = i + 1;
                });
                const ids = Array.from(cards).map(c => c.dataset.id);
                writeTextarea('ls-order-box', JSON.stringify(ids));
            }
        });
    }

    initSortable();
    if (!window._lsObserver) {
        window._lsObserver = new MutationObserver(initSortable);
        window._lsObserver.observe(document.body, {childList: true, subtree: true});
    }
}
"""

SORTABLE_CSS = """<style>
.ls-ghost { opacity:0.4; background:#4f46e5 !important; }
.ls-card:hover { border-color:#6b7280 !important; }
.ls-hidden { display: none !important; }
</style>"""


# ── Session ───────────────────────────────────────────────────────────────────

def _session_file(directory: str) -> Path:
    return Path(directory) / SESSION_NAME


def save_session(image_list: list, crop_data: dict, directory: str) -> str:
    if not directory.strip():
        return ""
    data = {
        "saved": datetime.now().isoformat(timespec="seconds"),
        "directory": directory,
        "image_list": image_list,
        "crop_data": crop_data,
    }
    _session_file(directory).write_text(json.dumps(data, indent=2))
    LAST_DIR_FILE.write_text(directory)
    return f"Session auto-saved  ({datetime.now().strftime('%H:%M:%S')})"


def load_session_for_dir(directory: str) -> dict | None:
    sf = _session_file(directory)
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text())
    except Exception:
        return None


def clear_session(directory: str = "") -> tuple:
    if directory:
        sf = _session_file(directory)
        if sf.exists():
            sf.unlink()
    if LAST_DIR_FILE.exists():
        LAST_DIR_FILE.unlink()
    empty_html = render_sortable_html([], {})
    return [], {}, None, "", empty_html, "None", "Session cleared"


# ── Image helpers ─────────────────────────────────────────────────────────────

def make_item(path: str) -> dict:
    return {"id": str(uuid4()), "path": path}


def get_effective_path(image_list: list, crop_data: dict, idx: int) -> str:
    return crop_data.get(str(idx), image_list[idx]["path"])


def thumb_b64(path: str) -> str:
    """Return a base64 JPEG data-URL thumbnail — works regardless of file-serving config."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        # Return a grey placeholder on error
        img = Image.new("RGB", THUMB_SIZE, (80, 80, 80))
        buf = BytesIO()
        img.save(buf, "JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_sortable_html(image_list: list, crop_data: dict) -> str:
    if not image_list:
        return SORTABLE_CSS + "<p style='color:#888;padding:20px'>Load a directory above.</p>"

    cards = []
    for i, item in enumerate(image_list):
        effective = get_effective_path(image_list, crop_data, i)
        name = Path(item["path"]).name
        badge = " ✓" if str(i) in crop_data else ""
        src = thumb_b64(effective)
        cards.append(f"""
        <div class="ls-card" data-id="{item['id']}" style="
            display:inline-block; margin:6px; cursor:grab; position:relative;
            border-radius:6px; overflow:hidden; vertical-align:top;
            border:2px solid #374151; user-select:none; outline:none;
        ">
            <img src="{src}" width="{THUMB_SIZE[0]}" height="{THUMB_SIZE[1]}"
                 style="display:block; object-fit:cover;" draggable="false">
            <div class="ls-num" style="
                position:absolute; top:5px; left:5px;
                background:rgba(0,0,0,0.75); color:white;
                padding:2px 8px; border-radius:12px; font-size:13px; font-weight:bold;
            ">{i + 1}</div>
            <div style="
                position:absolute; bottom:0; left:0; right:0;
                background:rgba(0,0,0,0.65); color:#e5e7eb;
                font-size:10px; padding:3px 5px;
                overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
            ">{name}{badge}</div>
        </div>""")

    return f"""{SORTABLE_CSS}
    <div id="ls-sortable" style="
        padding:10px; background:#111827; border-radius:8px;
        line-height:0; min-height:240px;
    ">{''.join(cards)}</div>"""


# ── Directory loading ─────────────────────────────────────────────────────────

def load_images_from_dir(directory: str):
    """Load a directory. Returns (image_list, crop_data, status_msg).

    Priority:
      1. .lifestages_session.json  — full session (list with copies + crops)
      2. order.json                — just file order (legacy / manual saves)
      3. alphabetical scan
    """
    d = Path(directory.strip())
    if not d.is_dir():
        return [], {}, "Directory not found"

    all_paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    # 1. Full session restore
    session = load_session_for_dir(directory)
    if session:
        image_list = session.get("image_list", [])
        crop_data  = session.get("crop_data", {})
        # Validate: drop entries whose source image no longer exists
        image_list = [it for it in image_list if Path(it["path"]).exists()]
        crop_data  = {k: v for k, v in crop_data.items()
                      if int(k) < len(image_list) and Path(v).exists()}
        # Append any new images not already in the session
        known = {it["path"] for it in image_list}
        for p in all_paths:
            if str(p) not in known:
                image_list.append(make_item(str(p)))
        ts = session.get("saved", "")
        return image_list, crop_data, f"Restored session from {ts} ({len(image_list)} images)"

    # 2. order.json fallback
    order_file = d / "order.json"
    if order_file.exists():
        try:
            saved_paths = json.loads(order_file.read_text())
            existing = [p for p in saved_paths if Path(p).exists()]
            saved_set = set(existing)
            for p in all_paths:
                if str(p) not in saved_set:
                    existing.append(str(p))
            return [make_item(p) for p in existing], {}, f"Loaded {len(existing)} images (order restored)"
        except Exception:
            pass

    # 3. Plain scan
    items = [make_item(str(p)) for p in all_paths]
    return items, {}, f"Loaded {len(items)} images"


# ── State operations ──────────────────────────────────────────────────────────

def _label_for(image_list: list, idx) -> str:
    if idx is None or not image_list:
        return "None"
    try:
        i = int(idx)
    except Exception:
        return "None"
    if i < 0 or i >= len(image_list):
        return "None"
    return f"[{i+1}] {Path(image_list[i]['path']).name}"


def apply_drag_order(new_order_json: str, image_list: list, crop_data: dict,
                     selected_idx, directory: str):
    if not new_order_json or not new_order_json.strip():
        return (image_list, crop_data, selected_idx,
                render_sortable_html(image_list, crop_data),
                _label_for(image_list, selected_idx), "")
    try:
        new_ids = json.loads(new_order_json)
    except Exception:
        return (image_list, crop_data, selected_idx,
                render_sortable_html(image_list, crop_data),
                _label_for(image_list, selected_idx), "")

    id_to_item  = {item["id"]: item for item in image_list}
    old_index   = {item["id"]: i    for i, item in enumerate(image_list)}

    new_list = [id_to_item[id_] for id_ in new_ids if id_ in id_to_item]
    new_crop = {}
    for new_i, id_ in enumerate(new_ids):
        old_i = old_index.get(id_)
        if old_i is not None and str(old_i) in crop_data:
            new_crop[str(new_i)] = crop_data[str(old_i)]

    # Track the previously-selected item by id so it follows the reorder
    new_selected = None
    if selected_idx is not None:
        try:
            old_i = int(selected_idx)
            if 0 <= old_i < len(image_list):
                old_id = image_list[old_i]["id"]
                for new_i, id_ in enumerate(new_ids):
                    if id_ == old_id:
                        new_selected = new_i
                        break
        except Exception:
            pass

    status = save_session(new_list, new_crop, directory)
    return (new_list, new_crop, new_selected,
            render_sortable_html(new_list, new_crop),
            _label_for(new_list, new_selected), status)


def copy_selected(image_list: list, crop_data: dict, selected_idx, directory: str):
    if selected_idx is None or not image_list:
        return image_list, crop_data, render_sortable_html(image_list, crop_data), "Nothing selected", ""
    idx = int(selected_idx)
    new_item = make_item(image_list[idx]["path"])
    new_list = list(image_list)
    new_list.insert(idx + 1, new_item)
    new_crop = {str(ki if ki <= idx else ki + 1): v for ki, v in
                ((int(k), v) for k, v in crop_data.items())}
    status = save_session(new_list, new_crop, directory)
    return new_list, new_crop, render_sortable_html(new_list, new_crop), f"Copied image {idx+1}", status


def remove_selected(image_list: list, crop_data: dict, selected_idx, directory: str):
    if selected_idx is None or not image_list:
        return image_list, crop_data, None, render_sortable_html(image_list, crop_data), "Nothing selected", "", "None"
    idx = int(selected_idx)
    new_list = [item for i, item in enumerate(image_list) if i != idx]
    new_crop = {}
    for k, v in crop_data.items():
        ki = int(k)
        if ki == idx:
            continue
        new_crop[str(ki - 1 if ki > idx else ki)] = v
    new_idx = min(idx, len(new_list) - 1) if new_list else None
    label   = (f"[{new_idx+1}] {Path(new_list[new_idx]['path']).name}"
               if new_idx is not None else "None")
    status  = save_session(new_list, new_crop, directory)
    return new_list, new_crop, new_idx, render_sortable_html(new_list, new_crop), f"Removed image {idx+1}", status, label


def rotate_image(image_list: list, crop_data: dict, selected_idx, degrees: int, directory: str):
    if selected_idx is None or not image_list:
        return crop_data, render_sortable_html(image_list, crop_data), "Nothing selected", ""
    idx = int(selected_idx)
    src = get_effective_path(image_list, crop_data, idx)
    img = Image.open(src).convert("RGB").rotate(-degrees, expand=True)
    rot_path = str(CROP_DIR / f"{Path(image_list[idx]['path']).stem}_rot{degrees}_{uuid4().hex[:6]}.jpg")
    img.save(rot_path, "JPEG", quality=95)
    new_crop = dict(crop_data)
    new_crop[str(idx)] = rot_path
    status = save_session(image_list, new_crop, directory)
    return new_crop, render_sortable_html(image_list, new_crop), f"Rotated {degrees}° → image {idx+1}", status


def save_crop_fn(image_list: list, crop_data: dict, selected_idx, cropped_image, directory: str):
    if selected_idx is None or cropped_image is None:
        return crop_data, "Nothing to save", render_sortable_html(image_list, crop_data), ""
    idx = int(selected_idx)
    img = (cropped_image.get("composite") or cropped_image.get("background")
           if isinstance(cropped_image, dict) else cropped_image)
    if img is None:
        return crop_data, "No image data", render_sortable_html(image_list, crop_data), ""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    crop_path = str(CROP_DIR / f"{Path(image_list[idx]['path']).stem}_crop_{uuid4().hex[:6]}.jpg")
    img.convert("RGB").save(crop_path, "JPEG", quality=95)
    new_crop = dict(crop_data)
    new_crop[str(idx)] = crop_path
    status = save_session(image_list, new_crop, directory)
    return new_crop, f"Saved crop for image {idx+1}", render_sortable_html(image_list, new_crop), status


def save_order_fn(image_list: list, directory: str):
    if not image_list or not directory.strip():
        return "No images loaded"
    d = Path(directory.strip())
    (d / "order.json").write_text(json.dumps([item["path"] for item in image_list], indent=2))
    return f"Saved order.json ({len(image_list)} images)"


def load_selected_for_crop(image_list: list, crop_data: dict, selected_idx):
    """Load the effective image (rotated/cropped if applicable) into the crop editor."""
    if not image_list or selected_idx is None:
        return None
    idx = int(selected_idx)
    if 0 <= idx < len(image_list):
        path = get_effective_path(image_list, crop_data, idx)
        img = Image.open(path).convert("RGB")
        return {"background": img, "layers": [], "composite": img}
    return None


def load_original_for_crop(image_list: list, selected_idx):
    """Load the original source image, discarding any crops/rotations."""
    if not image_list or selected_idx is None:
        return None
    idx = int(selected_idx)
    if 0 <= idx < len(image_list):
        img = Image.open(image_list[idx]["path"]).convert("RGB")
        return {"background": img, "layers": [], "composite": img}
    return None


def build_preview(image_list: list, crop_data: dict):
    return [
        (get_effective_path(image_list, crop_data, i),
         f"[{i+1}] {Path(item['path']).name}" + (" ✓" if str(i) in crop_data else ""))
        for i, item in enumerate(image_list)
    ]


# ── Video generation ──────────────────────────────────────────────────────────

def generate_clips(image_list, crop_data, prompt_template,
                   num_frames, height, width, strength_first, strength_last, seed,
                   progress=gr.Progress()):
    if len(image_list) < 2:
        return None, "Need at least 2 images"

    n_clips, clip_paths, log = len(image_list) - 1, [], []

    for i in range(n_clips):
        progress(i / n_clips, desc=f"Clip {i+1}/{n_clips}...")
        first = get_effective_path(image_list, crop_data, i)
        last  = get_effective_path(image_list, crop_data, i + 1)
        out   = str(OUTPUT_DIR / f"clip_{i:02d}.mp4")
        prompt = prompt_template.replace("{i}", str(i+1)).replace("{n}", str(n_clips))

        cmd = [
            LTX_PYTHON, "-m", "ltx_pipelines.ti2vid_two_stages",
            "--checkpoint-path", CHECKPOINT,
            "--distilled-lora", DISTILLED_LORA,
            "--quantization", "fp8-cast",
            "--prompt", prompt,
            "--num-frames", str(int(num_frames)),
            "--height", str(int(height)),
            "--width",  str(int(width)),
            "--seed",   str(int(seed) + i),
            "--image", first, "0", str(float(strength_first)),
            "--image", last,  str(int(num_frames) - 1), str(float(strength_last)),
            "--output-path", out,
        ]

        log.append(f"\n--- Clip {i+1}/{n_clips}: {Path(first).name} → {Path(last).name}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=LTX_CWD)
        if result.returncode != 0:
            log.append(f"ERROR:\n{result.stderr[-1500:]}")
            return None, "\n".join(log)
        log.append(f"OK → {out}")
        clip_paths.append(out)

    progress(1.0, desc="Stitching...")
    final = stitch_clips(clip_paths)
    log.append(f"\nFinal → {final}")
    return final, "\n".join(log)


def stitch_clips(clip_paths):
    import av
    out_path = str(OUTPUT_DIR / "final_lifestages.mp4")
    with av.open(clip_paths[0]) as src:
        vs = src.streams.video[0]
        fps, width, height = float(vs.average_rate), vs.width, vs.height

    with av.open(out_path, "w") as out_container:
        out_stream = out_container.add_stream("h264", rate=fps)
        out_stream.width, out_stream.height = width, height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18"}
        pts = 0
        for clip_path in clip_paths:
            with av.open(clip_path) as src:
                for frame in src.decode(src.streams.video[0]):
                    frame.pts = pts; pts += 1
                    for pkt in out_stream.encode(frame):
                        out_container.mux(pkt)
        for pkt in out_stream.encode():
            out_container.mux(pkt)
    return out_path


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Life Stages Video") as demo:

        # ── Shared state ───────────────────────────────────────────────────
        image_list_state = gr.State([])
        crop_data_state  = gr.State({})
        selected_idx_state = gr.State(None)
        dir_state        = gr.State("")

        # ── Header ────────────────────────────────────────────────────────
        gr.Markdown("# Life Stages Video Generator")
        with gr.Row():
            session_status = gr.Textbox(
                label="Session", interactive=False, scale=4,
                value="No saved session"
            )
            clear_session_btn = gr.Button("✕ New session", scale=1)

        # ── Tab 1: Sequence ────────────────────────────────────────────────
        with gr.Tab("1 · Sequence"):
            with gr.Row():
                dir_input = gr.Textbox(label="Image directory", scale=4)
                load_btn  = gr.Button("Load", scale=1, variant="primary")
            load_status = gr.Textbox(label="", interactive=False, show_label=False)

            gr.Markdown("**Drag** to reorder · **Click** to select · then use buttons below")

            sortable_html = gr.HTML(
                value=SORTABLE_CSS + "<p style='color:#888;padding:20px'>Load a directory above.</p>"
            )
            order_box  = gr.Textbox(
                elem_id="ls-order-box", elem_classes=["ls-hidden"],
                label="drag-order",
            )
            # Number must be visible — Gradio 6 doesn't render hidden components in the DOM
            select_box = gr.Number(
                elem_id="ls-select-box", label="Selected # (click card or type)",
                value=-1, precision=0, minimum=-1, scale=2,
            )

            with gr.Row():
                selected_label = gr.Textbox(label="Selected", interactive=False, scale=3)
                rot_cw_btn     = gr.Button("↻ Rotate CW",  scale=1)
                rot_ccw_btn    = gr.Button("↺ Rotate CCW", scale=1)
                copy_btn       = gr.Button("⧉ Copy",        scale=1)
                remove_btn     = gr.Button("✕ Remove",      scale=1)
                save_order_btn = gr.Button("💾 Save order", scale=1)

            action_status = gr.Textbox(label="", interactive=False, show_label=False)

        # ── Tab 2: Crop ────────────────────────────────────────────────────
        with gr.Tab("2 · Crop"):
            gr.Markdown("Select an image in **Sequence**, crop here, then **Save crop**.")
            with gr.Row():
                with gr.Column(scale=3):
                    crop_editor = gr.ImageEditor(
                        label="Crop image", type="pil",
                        transforms=("crop",), layers=False, eraser=False, brush=False,
                        height=600, elem_id="ls-crop-editor",
                    )
                with gr.Column(scale=1):
                    reload_crop_btn = gr.Button("↺ Reload original source")
                    save_crop_btn   = gr.Button("✓ Save crop", variant="primary")
                    crop_status     = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown("---")
                    refresh_preview_btn = gr.Button("Refresh preview")

            preview_gallery = gr.Gallery(
                label="Effective sequence (what LTX will use)",
                columns=8, height=160, allow_preview=True, object_fit="cover",
            )

        # ── Tab 3: Generate ────────────────────────────────────────────────
        with gr.Tab("3 · Generate"):
            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(
                        label="Prompt template  ({i} = clip #, {n} = total clips)",
                        value="A smooth cinematic transition showing a person aging gracefully. "
                              "Natural lighting, consistent identity.",
                        lines=4,
                    )
                with gr.Column():
                    num_frames     = gr.Slider(49, 241, value=73, step=8, label="Frames per clip (8K+1)")
                    with gr.Row():
                        height_sl  = gr.Slider(256, 1024, value=512, step=32, label="Height")
                        width_sl   = gr.Slider(256, 1536, value=768, step=32, label="Width")
                    with gr.Row():
                        str_first  = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="First frame strength")
                        str_last   = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Last frame strength")
                    seed = gr.Number(value=42, label="Base seed", precision=0)

            gen_btn     = gr.Button("Generate all clips + stitch", variant="primary", size="lg")
            gen_log     = gr.Textbox(label="Log", lines=15, interactive=False)
            final_video = gr.Video(label="Final video")

        # ── Events ────────────────────────────────────────────────────────

        # Restore session on startup using last-used directory
        def on_app_load():
            if not LAST_DIR_FILE.exists():
                return [], {}, None, "", render_sortable_html([], {}), "None", "No saved session"
            last_dir = LAST_DIR_FILE.read_text().strip()
            if not last_dir or not Path(last_dir).is_dir():
                return [], {}, None, "", render_sortable_html([], {}), "None", "No saved session"
            il, cd, msg = load_images_from_dir(last_dir)
            return il, cd, None, last_dir, render_sortable_html(il, cd), "None", msg

        demo.load(
            on_app_load, outputs=[
                image_list_state, crop_data_state, selected_idx_state,
                dir_state, sortable_html, selected_label, session_status
            ]
        )
        demo.load(fn=None, js=SORTABLE_SETUP_JS)

        # Load directory — restores full session if .lifestages_session.json exists
        def on_load(directory):
            items, crop_data, msg = load_images_from_dir(directory)
            html = render_sortable_html(items, crop_data)
            ss = save_session(items, crop_data, directory) if items else "No images"
            return items, crop_data, None, directory, html, msg, "None", ss

        load_btn.click(on_load, inputs=[dir_input],
            outputs=[image_list_state, crop_data_state, selected_idx_state,
                     dir_state, sortable_html, load_status, selected_label, session_status])

        # Clear session
        clear_session_btn.click(
            clear_session,
            inputs=[dir_state],
            outputs=[image_list_state, crop_data_state, selected_idx_state,
                     dir_state, sortable_html, selected_label, session_status]
        )

        # Drag reorder (also remaps selected_idx to follow the moved item)
        order_box.change(
            apply_drag_order,
            inputs=[order_box, image_list_state, crop_data_state, selected_idx_state, dir_state],
            outputs=[image_list_state, crop_data_state, selected_idx_state,
                     sortable_html, selected_label, session_status],
        )

        # Click select (or manual number entry)
        def on_click_select(idx_val, image_list):
            if idx_val is None or not image_list:
                return None, "None"
            try:
                idx = int(idx_val)
                if idx < 0 or idx >= len(image_list):
                    return None, "None"
                return idx, f"[{idx+1}] {Path(image_list[idx]['path']).name}"
            except Exception:
                return None, "None"

        select_box.input(on_click_select, inputs=[select_box, image_list_state],
                         outputs=[selected_idx_state, selected_label])

        # JS that activates the Crop tool inside the ImageEditor whenever a new image loads.
        ACTIVATE_CROP_JS = """
        async () => {
            const editor = document.getElementById('ls-crop-editor');
            if (!editor) return;
            for (let i = 0; i < 30; i++) {
                const btn = editor.querySelector('button[aria-label="Crop"]');
                if (btn) { btn.click(); return; }
                await new Promise(r => setTimeout(r, 100));
            }
        }
        """

        # Rotate (also reloads the crop editor so rotation is reflected on the Crop tab)
        rot_cw_btn.click(
            lambda il, cd, idx, d: rotate_image(il, cd, idx, 90, d),
            inputs=[image_list_state, crop_data_state, selected_idx_state, dir_state],
            outputs=[crop_data_state, sortable_html, action_status, session_status],
        ).then(
            load_selected_for_crop,
            inputs=[image_list_state, crop_data_state, selected_idx_state],
            outputs=[crop_editor],
        ).then(fn=None, js=ACTIVATE_CROP_JS)
        rot_ccw_btn.click(
            lambda il, cd, idx, d: rotate_image(il, cd, idx, -90, d),
            inputs=[image_list_state, crop_data_state, selected_idx_state, dir_state],
            outputs=[crop_data_state, sortable_html, action_status, session_status],
        ).then(
            load_selected_for_crop,
            inputs=[image_list_state, crop_data_state, selected_idx_state],
            outputs=[crop_editor],
        ).then(fn=None, js=ACTIVATE_CROP_JS)

        # Copy
        copy_btn.click(copy_selected,
            inputs=[image_list_state, crop_data_state, selected_idx_state, dir_state],
            outputs=[image_list_state, crop_data_state, sortable_html, action_status, session_status])

        # Remove
        remove_btn.click(remove_selected,
            inputs=[image_list_state, crop_data_state, selected_idx_state, dir_state],
            outputs=[image_list_state, crop_data_state, selected_idx_state,
                     sortable_html, action_status, session_status, selected_label])

        # Save order to directory
        save_order_btn.click(save_order_fn,
            inputs=[image_list_state, dir_state], outputs=[action_status])

        # Crop tab — auto-loads effective image (with rotation/crop applied)
        # Auto-activates the Crop tool after each load.
        reload_crop_btn.click(load_original_for_crop,
            inputs=[image_list_state, selected_idx_state], outputs=[crop_editor]
        ).then(fn=None, js=ACTIVATE_CROP_JS)
        selected_idx_state.change(load_selected_for_crop,
            inputs=[image_list_state, crop_data_state, selected_idx_state], outputs=[crop_editor]
        ).then(fn=None, js=ACTIVATE_CROP_JS)
        save_crop_btn.click(save_crop_fn,
            inputs=[image_list_state, crop_data_state, selected_idx_state, crop_editor, dir_state],
            outputs=[crop_data_state, crop_status, sortable_html, session_status])

        refresh_preview_btn.click(build_preview,
            inputs=[image_list_state, crop_data_state], outputs=[preview_gallery])

        # Generate
        gen_btn.click(generate_clips,
            inputs=[image_list_state, crop_data_state, prompt_box,
                    num_frames, height_sl, width_sl, str_first, str_last, seed],
            outputs=[final_video, gen_log])

    return demo


if __name__ == "__main__":
    import sys
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False,
        allowed_paths=[str(Path.home())] + sys.argv[1:],
    )
