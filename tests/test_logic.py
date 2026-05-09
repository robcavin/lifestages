"""Unit tests for lifestages app logic (no UI)."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_img_dir(tmp_path):
    """Directory with 3 small test images."""
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        img = Image.new("RGB", (100, 150), color=(120, 80, 40))
        img.save(tmp_path / name)
    return tmp_path


@pytest.fixture
def three_items(tmp_img_dir):
    items, _, _ = app.load_images_from_dir(str(tmp_img_dir))
    return items


# ── make_item ─────────────────────────────────────────────────────────────────

def test_make_item_has_unique_ids():
    a = app.make_item("/some/path.jpg")
    b = app.make_item("/some/path.jpg")
    assert a["id"] != b["id"]
    assert a["path"] == b["path"]


# ── load_images_from_dir ──────────────────────────────────────────────────────

def test_load_images_from_dir(tmp_img_dir):
    items, crop_data, msg = app.load_images_from_dir(str(tmp_img_dir))
    assert len(items) == 3
    assert all("id" in it and "path" in it for it in items)
    assert "3" in msg


def test_load_images_missing_dir():
    items, crop_data, msg = app.load_images_from_dir("/nonexistent/path")
    assert items == []
    assert "not found" in msg.lower()


def test_load_images_restores_saved_order(tmp_img_dir):
    paths = sorted(str(p) for p in tmp_img_dir.iterdir())
    reversed_paths = list(reversed(paths))
    (tmp_img_dir / "order.json").write_text(json.dumps(reversed_paths))

    items, crop_data, msg = app.load_images_from_dir(str(tmp_img_dir))
    assert [it["path"] for it in items] == reversed_paths
    assert "restored" in msg.lower()


# ── get_effective_path ────────────────────────────────────────────────────────

def test_get_effective_path_returns_original_when_no_crop(three_items):
    result = app.get_effective_path(three_items, {}, 0)
    assert result == three_items[0]["path"]


def test_get_effective_path_returns_crop_when_present(three_items):
    crop_data = {"1": "/some/crop.jpg"}
    result = app.get_effective_path(three_items, crop_data, 1)
    assert result == "/some/crop.jpg"


# ── apply_drag_order ──────────────────────────────────────────────────────────

def test_apply_drag_order_reorders(three_items):
    original_ids = [it["id"] for it in three_items]
    new_order = [original_ids[2], original_ids[0], original_ids[1]]
    new_list, new_crop, _html, _status = app.apply_drag_order(
        json.dumps(new_order), three_items, {}, ""
    )
    assert [it["id"] for it in new_list] == new_order


def test_apply_drag_order_preserves_crop_assignment(three_items):
    ids = [it["id"] for it in three_items]
    crop_data = {"0": "/crop_for_a.jpg"}
    # Move item 0 to position 2
    new_order = [ids[1], ids[2], ids[0]]
    new_list, new_crop, _html, _status = app.apply_drag_order(
        json.dumps(new_order), three_items, crop_data, ""
    )
    # Crop that was on item 0 should now follow it to position 2
    assert new_crop.get("2") == "/crop_for_a.jpg"
    assert "0" not in new_crop


def test_apply_drag_order_invalid_json_returns_unchanged(three_items):
    new_list, new_crop, _html, _status = app.apply_drag_order(
        "not-json", three_items, {}, ""
    )
    assert new_list == three_items


# ── copy_selected ─────────────────────────────────────────────────────────────

def test_copy_selected_inserts_after(three_items):
    new_list, _, _html, msg, _status = app.copy_selected(three_items, {}, 1, "")
    assert len(new_list) == 4
    assert new_list[2]["path"] == three_items[1]["path"]
    assert new_list[2]["id"] != three_items[1]["id"]


def test_copy_selected_new_id_is_unique(three_items):
    all_ids_before = {it["id"] for it in three_items}
    new_list, _, _html, _msg, _status = app.copy_selected(three_items, {}, 0, "")
    assert new_list[1]["id"] not in all_ids_before


def test_copy_selected_shifts_crop_indices(three_items):
    crop_data = {"2": "/crop_c.jpg"}
    new_list, new_crop, _html, _msg, _status = app.copy_selected(three_items, crop_data, 0, "")
    # Original item 2 is now at index 3 (copy inserted at 1)
    assert new_crop.get("3") == "/crop_c.jpg"
    assert "2" not in new_crop


def test_copy_selected_nothing_selected(three_items):
    new_list, new_crop, _html, msg, _status = app.copy_selected(three_items, {}, None, "")
    assert len(new_list) == len(three_items)
    assert "nothing" in msg.lower()


# ── remove_selected ───────────────────────────────────────────────────────────

def test_remove_selected_removes_correct_item(three_items):
    path_to_remove = three_items[1]["path"]
    new_list, _, _, _, _, _, _ = app.remove_selected(three_items, {}, 1, "")
    assert len(new_list) == 2
    assert all(it["path"] != path_to_remove for it in new_list)


def test_remove_selected_shifts_crop_indices(three_items):
    crop_data = {"2": "/crop_c.jpg"}
    new_list, new_crop, _, _, _, _, _ = app.remove_selected(three_items, crop_data, 0, "")
    # Item 2 shifts to index 1 after removing item 0
    assert new_crop.get("1") == "/crop_c.jpg"


# ── rotate_image ──────────────────────────────────────────────────────────────

def test_rotate_image_creates_crop(three_items, tmp_path):
    app.CROP_DIR.mkdir(exist_ok=True)
    new_crop, _, msg, _ = app.rotate_image(three_items, {}, 0, 90, "")
    assert "0" in new_crop
    assert Path(new_crop["0"]).exists()
    assert "90" in msg


def test_rotate_image_cw_changes_dimensions(three_items):
    img_before = Image.open(three_items[0]["path"])
    w, h = img_before.size
    app.rotate_image(three_items, {}, 0, 90, "")
    new_crop_path = app.rotate_image(three_items, {}, 0, 90, "")[0]["0"]
    img_after = Image.open(new_crop_path)
    # 90° rotation swaps width and height
    assert img_after.size == (h, w)


def test_rotate_image_nothing_selected(three_items):
    new_crop, _, msg, _ = app.rotate_image(three_items, {}, None, 90, "")
    assert new_crop == {}
    assert "nothing" in msg.lower()


# ── session ───────────────────────────────────────────────────────────────────

def test_session_roundtrip(three_items, tmp_path):
    directory = str(tmp_path)
    crop_data = {"0": "/crop.jpg"}
    app.save_session(three_items, crop_data, directory)

    loaded = app.load_session_for_dir(directory)
    assert loaded is not None
    assert loaded["directory"] == directory
    assert loaded["image_list"] == three_items
    assert loaded["crop_data"] == crop_data


def test_load_session_returns_none_when_missing(tmp_path):
    assert app.load_session_for_dir(str(tmp_path)) is None


def test_clear_session_removes_file(tmp_path):
    directory = str(tmp_path)
    app.save_session([], {}, directory)
    assert (tmp_path / app.SESSION_NAME).exists()
    app.clear_session(directory)
    assert not (tmp_path / app.SESSION_NAME).exists()


# ── render_sortable_html ──────────────────────────────────────────────────────

def test_render_sortable_html_contains_all_ids(three_items):
    html = app.render_sortable_html(three_items, {})
    for item in three_items:
        assert item["id"] in html


def test_render_sortable_html_empty():
    html = app.render_sortable_html([], {})
    assert "ls-sortable" not in html


def test_render_sortable_html_shows_crop_badge(three_items):
    crop_data = {"1": three_items[1]["path"]}  # reuse same path for test
    html = app.render_sortable_html(three_items, crop_data)
    # The crop badge ✓ should appear once (for index 1)
    assert html.count("✓") == 1
