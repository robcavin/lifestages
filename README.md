# Life Stages Video Generator

Organise portrait photos by life stage, crop/rotate each one, then generate
smooth transition clips between them using LTX-2.3 and stitch into a single video.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

## Video generation (optional)

The **Sequence** and **Crop** tabs work standalone. The **Generate** tab requires
a working [LTX-2 installation](https://github.com/Lightricks/LTX-Video) with the
22B dev checkpoint. Point the app at it via environment variables:

```bash
export LTX_PYTHON=/path/to/ltx-repo/.venv/bin/python
export LTX_CWD=/path/to/ltx-repo
export LTX_CHECKPOINT=~/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots/main/ltx-2.3-22b-dev.safetensors
export LTX_DISTILLED_LORA=~/.cache/huggingface/hub/models--Lightricks--LTX-Video/snapshots/main/ltx-2.3-22b-distilled-lora.safetensors
python app.py
```

Or copy `.env.example` to `.env` and fill in the values (if you add
`python-dotenv` to requirements).

## Workflow

1. **Sequence tab** — paste the path to your image folder and click **Load**.
   Drag cards to reorder. Click a card to select it.
2. **Crop tab** — the selected image loads automatically. Crop, then **Save crop**.
   Use **↺ Reload original source** to start the crop fresh from the original.
   Rotate CW/CCW from the Sequence tab before cropping if needed.
3. **Generate tab** — set prompt, resolution, frame count, and click
   **Generate all clips + stitch**.

Session is saved automatically to `.lifestages_session.json` inside your image
directory so all your work is restored next time you load the same folder.

## Tests

```bash
pip install -r requirements-dev.txt
playwright install chromium

# Unit tests (no browser needed)
python -m pytest tests/test_logic.py -v

# UI tests (requires the app to be running on :7860)
python app.py &
python -m pytest tests/test_ui.py -v
```
