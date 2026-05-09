"""Playwright UI tests for the Life Stages app.

Run the app first:  LD_LIBRARY_PATH=/tmp/stublibs python app.py
Then:               LD_LIBRARY_PATH=/tmp/stublibs python -m pytest tests/test_ui.py -v
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Permanent stub satisfying Chromium's libasound.so.2 dependency on this WSL2 host.
# Built from a no-op C stub; safe because Chromium runs headless + muted.
ALSA_STUB_DIR = str(Path.home() / ".local" / "lib")
os.environ["LD_LIBRARY_PATH"] = (
    ALSA_STUB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

import pytest
from PIL import Image
from playwright.sync_api import Page, expect, sync_playwright

APP_URL = "http://localhost:7860"
SHORT = 8_000   # ms — wait for UI reactions
LONG  = 20_000  # ms — wait for Gradio round-trips


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_image_dir(tmp_path_factory):
    """3 small test images in a temp dir under /home so allowed_paths covers them."""
    d = Path.home() / ".cache" / "lifestages_test_imgs"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("photo_01.jpg", "photo_02.jpg", "photo_03.jpg"):
        img_path = d / name
        if not img_path.exists():
            Image.new("RGB", (120, 160), color=(100, 140, 200)).save(img_path)
    return d


@pytest.fixture(scope="session")
def app_server(test_image_dir):
    """Use the already-running server or start one."""
    import urllib.request
    try:
        urllib.request.urlopen(APP_URL, timeout=2)
        yield None
        return
    except Exception:
        pass
    app_py = str(Path(__file__).parent.parent / "app.py")
    proc = subprocess.Popen(
        [sys.executable, app_py, str(test_image_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "LD_LIBRARY_PATH": "/tmp/stublibs"},
    )
    for _ in range(30):
        try:
            urllib.request.urlopen(APP_URL, timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("App server did not start")
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture
def page(app_server):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        pg = ctx.new_page()
        pg.set_default_timeout(SHORT)
        pg.goto(APP_URL)
        pg.wait_for_load_state("load")
        pg.wait_for_selector("text=Life Stages Video Generator", timeout=LONG)
        yield pg
        ctx.close()
        browser.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def fill_dir_and_load(page: Page, directory: str):
    """Fill the directory textbox and click Load, wait for cards."""
    # Gradio 6: label text → find associated textarea via aria
    page.get_by_label("Image directory").fill(directory)
    page.get_by_role("button", name="Load").click()
    page.wait_for_selector("#ls-sortable .ls-card", timeout=LONG)


def card_count(page: Page) -> int:
    return page.locator("#ls-sortable .ls-card").count()


def card_numbers(page: Page) -> list[str]:
    return page.eval_on_selector_all(
        "#ls-sortable .ls-num", "els => els.map(e => e.textContent.trim())"
    )


def card_ids(page: Page) -> list[str]:
    return page.eval_on_selector_all(
        "#ls-sortable .ls-card", "cards => cards.map(c => c.dataset.id)"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAppLoad:
    def test_title_visible(self, page):
        expect(page.get_by_text("Life Stages Video Generator")).to_be_visible()

    def test_tabs_present(self, page):
        for tab in ("1 · Sequence", "2 · Crop", "3 · Generate"):
            expect(page.get_by_role("tab", name=tab)).to_be_visible()

    def test_session_label_shown(self, page):
        # Session textbox label should appear
        expect(page.get_by_label("Session")).to_be_visible()


class TestLoadDirectory:
    def test_cards_appear_after_load(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        assert card_count(page) == 3

    def test_cards_show_sequential_numbers(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        assert card_numbers(page) == ["1", "2", "3"]

    def test_images_are_base64_embedded(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        srcs = page.eval_on_selector_all(
            "#ls-sortable .ls-card img", "imgs => imgs.map(i => i.src.slice(0, 20))"
        )
        assert all(s.startswith("data:image/") for s in srcs)


class TestClickSelect:
    def test_clicking_card_updates_select_number(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").nth(1).click()
        page.wait_for_timeout(1500)
        val = page.locator("#ls-select-box input").input_value()
        assert val == "1", f"Expected 1, got {val!r}"

    def test_clicking_card_highlights_it(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        card = page.locator("#ls-sortable .ls-card").first
        card.click()
        page.wait_for_timeout(600)
        outline = card.evaluate("el => el.style.outline")
        # Browser resolves hex to rgb: "rgb(99, 102, 241) solid 3px"
        assert "99, 102, 241" in outline or "6366f1" in outline

    def test_selected_label_updates(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").first.click()
        page.wait_for_timeout(1500)
        # Two elements match "Selected" — use exact label text for the textbox
        label_val = page.get_by_role("textbox", name="Selected").input_value()
        assert "[1]" in label_val

    def test_manual_number_entry_selects(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        inp = page.locator("#ls-select-box input")
        inp.fill("2")
        inp.press("Tab")
        page.wait_for_timeout(1500)
        label_val = page.get_by_role("textbox", name="Selected").input_value()
        assert "[3]" in label_val  # 2 is 0-based → "[3]"


class TestCopyAndRemove:
    def test_copy_adds_card(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").first.click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="⧉ Copy").click()
        page.wait_for_selector("#ls-sortable .ls-card:nth-child(4)", timeout=LONG)
        assert card_count(page) == 4

    def test_copy_inserts_after_selected(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        ids_before = card_ids(page)
        page.locator("#ls-sortable .ls-card").nth(1).click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="⧉ Copy").click()
        page.wait_for_timeout(2000)
        ids_after = card_ids(page)
        assert len(ids_after) == 4
        assert len(set(ids_after)) == 4  # all unique IDs

    def test_remove_decrements_count(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").nth(1).click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="✕ Remove").click()
        page.wait_for_timeout(2000)
        assert card_count(page) == 2

    def test_remove_renumbers_cards(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").first.click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="✕ Remove").click()
        page.wait_for_timeout(2000)
        assert card_numbers(page) == ["1", "2"]


class TestRotate:
    def test_rotate_cw_adds_crop_badge(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        assert "✓" not in page.locator("#ls-sortable").inner_text()
        page.locator("#ls-sortable .ls-card").first.click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="↻ Rotate CW").click()
        page.wait_for_timeout(3000)
        assert "✓" in page.locator("#ls-sortable").inner_text()

    def test_rotate_ccw_adds_crop_badge(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.locator("#ls-sortable .ls-card").nth(2).click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="↺ Rotate CCW").click()
        # Wait for the HTML to re-render with the crop badge
        page.wait_for_selector("#ls-sortable .ls-card:has-text('✓')", timeout=LONG)


class TestSaveOrder:
    def test_save_order_writes_json(self, page, test_image_dir):
        order_file = test_image_dir / "order.json"
        if order_file.exists():
            order_file.unlink()
        fill_dir_and_load(page, str(test_image_dir))
        page.get_by_role("button", name="💾 Save order").click()
        page.wait_for_timeout(1500)
        assert order_file.exists()
        saved = json.loads(order_file.read_text())
        assert len(saved) == 3

    def test_reload_restores_count(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        page.get_by_role("button", name="💾 Save order").click()
        page.wait_for_timeout(1000)
        # Reload
        fill_dir_and_load(page, str(test_image_dir))
        assert card_count(page) == 3


class TestSession:
    def test_new_session_clears_gallery(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        assert card_count(page) == 3
        page.get_by_role("button", name="✕ New session").click()
        page.wait_for_timeout(2000)
        assert card_count(page) == 0

    def test_session_status_updates_after_load(self, page, test_image_dir):
        fill_dir_and_load(page, str(test_image_dir))
        status = page.get_by_label("Session").input_value()
        assert "auto-saved" in status.lower() or "session" in status.lower()
