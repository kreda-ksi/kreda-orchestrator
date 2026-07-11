import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from kreda.models.events import AlignedSegment


def load_grid(grid_path: Path) -> tuple[dict, dict]:
    with open(grid_path, "r") as f:
        raw_data = json.load(f)

    header = raw_data.get("_stream_metadata", {"grid_dimensions": [12, 6]})

    normalized_frames = {}
    for raw_path, meta in raw_data.get("frames", {}).items():
        filename = Path(raw_path).name
        normalized_frames[filename] = meta

    return header, normalized_frames


def has_new_bg_content(
    curr_img_path: Path,
    prev_img_path: Path,
    occupancy_grid: list[list[int]],
    grid_dimensions: list[int],
    diff_threshold_pixels: int = 500,
) -> bool:
    curr_img = cv2.imread(str(curr_img_path), cv2.IMREAD_GRAYSCALE)
    prev_img = cv2.imread(str(prev_img_path), cv2.IMREAD_GRAYSCALE)

    if curr_img is None or prev_img is None:
        return True

    img_h, img_w = curr_img.shape
    grid_cols, grid_rows = grid_dimensions[0], grid_dimensions[1]
    cell_h = img_h // grid_rows
    cell_w = img_w // grid_cols

    # 255 = actual chalkboard data, 0 = ignore (lecturer is/recently was here)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if occupancy_grid[r][c] == 0:
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                mask[y1:y2, x1:x2] = 255

    diff = cv2.absdiff(curr_img, prev_img)

    # block out motion
    masked_diff = cv2.bitwise_and(diff, mask)

    # ignore minor lighting changes/compression artifacts
    _, thresh = cv2.threshold(masked_diff, 30, 255, cv2.THRESH_BINARY)

    return bool(np.count_nonzero(thresh) > diff_threshold_pixels)


def run(
    aligned_segments: list[AlignedSegment], run_path: Path, grid_file: Path
) -> list[AlignedSegment]:
    header, frames = load_grid(run_path / grid_file)
    grid_dimensions = header.get("grid_dimensions", [12, 6])

    curated_segments = []
    last_kept_image_path: Optional[Path] = None

    for segment in aligned_segments:
        frame = frames.get(segment.filename)
        if not frame:
            curated_segments.append(segment)
            continue

        grid = frame.get("occupancy_grid")
        img_path = run_path / segment.filename

        total_cells = grid_dimensions[0] * grid_dimensions[1]
        occluded_cells = sum(1 for row in grid for cell in row if cell > 0)
        occlusion_ratio = occluded_cells / total_cells

        if segment.event_type == "SAVE_final" or occlusion_ratio < 0.10:
            curated_segments.append(segment)
            last_kept_image_path = img_path
            continue

        if last_kept_image_path is not None:
            if has_new_bg_content(
                img_path, last_kept_image_path, grid, grid_dimensions
            ):
                curated_segments.append(segment)
                last_kept_image_path = img_path
            else:
                print(f"Dropped {segment.filename}.")
                # append dropped segment's transcript to previous to not lose audio
                if curated_segments:
                    curated_segments[-1].transcript_chunk += (
                        " " + segment.transcript_chunk
                    )
        else:  # first image in sequence
            curated_segments.append(segment)
            last_kept_image_path = img_path

    return curated_segments
