import cv2

# ============================================================
# ChArUco board parameters
# ============================================================

squares_x = 7
squares_y = 5

square_length_m = 0.040   # 40 mm
marker_length_m = 0.030   # 30 mm

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length_m,
    marker_length_m,
    dictionary
)

# ============================================================
# Generate image
#
# Physical board:
#   7 * 40 mm = 280 mm
#   5 * 40 mm = 200 mm
#
# Use 10 pixels/mm:
#   2800 x 2000 px
# ============================================================

width_px = 2800
height_px = 2000

img = board.generateImage(
    (width_px, height_px),
    marginSize=0,
    borderBits=1
)

output_path = "charuco_7x5_40mm_30mm.png"

cv2.imwrite(output_path, img)

print("Saved:", output_path)
print("Image resolution:", img.shape[1], "x", img.shape[0])
print("Print pattern size: 280 mm x 200 mm")
print("Square size: 40 mm")
print("Marker size: 30 mm")
print("Dictionary: DICT_4X4_50")