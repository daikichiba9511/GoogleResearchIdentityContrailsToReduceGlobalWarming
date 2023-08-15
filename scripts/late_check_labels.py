"""

## Reference

- https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430749?rvi=1
"""
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

h, w = 8, 8

mask = np.zeros((h, w), dtype=np.uint8)
pts = np.array([[1, 5], [2, 3], [6, 4], [3, 1]])
pts_corrected = pts + 0.5

pts_reshaped = pts.reshape((-1, 1, 2)).astype(np.int32)

# cv2の座標系に変換(左上原点 -> 中央原点)
cv2.fillPoly(img=mask, pts=[pts_reshaped], color=(255,))

fig, ax = plt.subplots()
assert isinstance(ax, plt.Axes) and isinstance(fig, plt.Figure)

# extent = [left, right, bottom, top]
ax.imshow(mask, cmap="gray", extent=[0, w, h, 0])
polygon = patches.Polygon(pts, edgecolor="red", facecolor="none", label="polygon")
polygon_corrected = patches.Polygon(
    pts_corrected, edgecolor="blue", facecolor="none", label="polygon_corrected"
)
ax.add_patch(polygon)
ax.add_patch(polygon_corrected)
ax.legend()
ax.set_title("Result of cv2.fillPoly")
fig.savefig("fillpoly.png", bbox_inches="tight")
