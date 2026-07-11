import cv2
import numpy as np
from kreda.pipeline.assembler import has_new_bg_content


def test_asmked_bg_differencing(tmp_path):
    img1 = np.zeros((60, 120), dtype=np.uint8)
    img2 = np.zeros((60, 120), dtype=np.uint8)

    grid = [[0 for _ in range(12)] for _ in range(6)]

    grid[0][0] = 1
    img2[5, 5] = 255

    p1 = tmp_path / "frame1.png"
    p2 = tmp_path / "frame2.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    has_diff = has_new_bg_content(p2, p1, grid, [12, 6], 0)
    assert has_diff is False

    img2[5, 5] = 0
    img2[25, 25] = 255
    cv2.imwrite(str(p2), img2)

    has_diff = has_new_bg_content(p2, p1, grid, [12, 6], 0)
    assert has_diff is True
