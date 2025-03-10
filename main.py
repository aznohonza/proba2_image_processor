from typing import List, Tuple
import numpy as np
import imageio
import cv2
import os
import re


def extract_sort_key(file_path: str) -> Tuple[str, int]:
    match = re.search(r"SWAP_(\d+T\d+Z)(?:-(\d+))?", file_path)
    if match:
        timestamp, numeric_suffix = match.groups()
        return (timestamp, int(numeric_suffix) if numeric_suffix else -1)
    return (None, None)

def calculate_image_difference(reference_image: np.ndarray, image: np.ndarray) -> float:
    return cv2.norm(reference_image, cv2.bitwise_and(image, image, mask=MASK), cv2.NORM_L2)

def correct_rotation(reference_image: np.ndarray, image: np.ndarray) -> np.ndarray:
    rotated_images = [image]
    for _ in range(3):
        rotated_images.append(cv2.rotate(rotated_images[-1], cv2.ROTATE_90_CLOCKWISE))
    rotated_image = sorted(rotated_images, key=lambda x: calculate_image_difference(reference_image, x))[0]
    return rotated_image

def translate_image(image: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(image, translation_matrix, (width, height))

def correct_offset(reference_image: np.ndarray, img: np.ndarray, offset: int, step: int) -> np.ndarray:
    offset_images = []
    for x in range(-offset, offset+1, step):
        for y in range(-offset, offset+1, step):
            offset_images.append(translate_image(img, x, y))
    offset_image = sorted(offset_images, key=lambda x: calculate_image_difference(reference_image, x))[0]
    return offset_image

def compute_tenengrad(img: np.ndarray) -> int:
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = round(np.mean(sobel_x**2 + sobel_y**2))
    return tenengrad

def enhance_image(image: np.ndarray, enhancment: List) -> np.ndarray:
    brightness = enhancment[0]
    contrast = enhancment[1]
    tint = (enhancment[2], enhancment[3], enhancment[4])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2].astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    image_float = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    tinted_image = image_float * np.array(tint)
    mean_value = np.mean(tinted_image, axis=(0, 1))
    adjusted_image = np.clip((tinted_image - mean_value) * contrast + mean_value, 0, 1)
    return (adjusted_image * 255).astype(np.uint8)


# Settings
INPUT_DIR = "SWAP/"           # SWAP folder from proba2 dump
OUTPUT_FILE = "output.gif"    # output file (.gif or .webp)
MS_PER_FRAME = 100            # time (ms) each frame is displayed in the output
EXTRA_PROCESSING = False      # only set to True if the output is not centered, normally shouldn't be needed
ENHANCEMENT_NAME = "official" # set to None to get output without enhancements
SHOW_IMG_REALTIME = False     # will show last proccessed frame in a new window, set to True only if you installed "opencv-python" ("opencv-python-headless" doesn't support GUIs)

# Tresholds
BLURRYNESS_TRESHOLD = 500
SQUARENESS_TRESHOLD = 1085

# Define enhancements (brightness, contrast, red_coeff, green_coeff, blue_coeff)
ENHANCEMENTS = {
    'official': [79.63933029251594, 2.9539919324832264, 0.8714331298994634, 0.5378500771455333, 0.03210348184168943],
    'official_blue': [79.63933029251594, 2.9539919324832264, 0.03210348184168943, 0.5378500771455333, 0.8714331298994634],
    'orange': [20.0, 4.0, 0.93, 0.44, 0.01],
    'blue': [20.0, 4.0, 0.11, 0.50, 0.98],
    'green': [20.0, 4.0, 0.44, 0.93, 0.01],
    'pink': [20.0, 4.0, 0.93, 0.01, 0.44],
    'magenta': [20.0, 4.0, 0.95, 0.25, 0.85],
    'purple': [20.0, 4.0, 0.45, 0.11, 0.51],
}

# Load calibration image (thanks to @soplica.pl on discord for providing this image)
calibration_img = cv2.imread(".calibration.png")

# Compute circular mask for SWAP images to save on computing image scores
h, w = calibration_img.shape[:2]
MASK = np.zeros((h, w), dtype=np.uint8)
cv2.circle(MASK, (w // 2, h // 2), min(h, w) // 3, 255, -1)

# Apply the mask
calibration_img = cv2.bitwise_and(calibration_img, calibration_img, mask=MASK)

# Get image paths
image_paths = [os.path.join(INPUT_DIR, file) for file in os.listdir(INPUT_DIR) if "SWAP_" in file and file.endswith((".png", ".jpg"))]
image_paths = sorted(image_paths, key=extract_sort_key)

# Prepare list to store corrected frames
output_frames = []

# Main loop
for i, img_path in enumerate(image_paths):
    # Print progress
    print(f"INFO   processing   {i+1: >2}/{len(image_paths)}   {os.path.basename(img_path):<28}", end="", flush=True)

    # Load image
    img = cv2.imread(img_path)

    # Skip bad images
    tenengrad = compute_tenengrad(img)
    if tenengrad < BLURRYNESS_TRESHOLD:
        print(f"   skipping cuz blurry ({tenengrad})")
        continue
    if tenengrad > SQUARENESS_TRESHOLD:
        print(f"   skipping cuz squarey ({tenengrad})")
        continue

    # Rotate and center images in 3 steps
    # step 0
    if EXTRA_PROCESSING == True:
        img = correct_rotation(calibration_img, img)
        img = correct_offset(calibration_img, img, 50, 25)
    # step 1
    img = correct_rotation(calibration_img, img)      #  1 time
    img = correct_offset(calibration_img, img, 27, 9) # 49 times
    # step 2
    img = correct_rotation(calibration_img, img)      #  1 time
    img = correct_offset(calibration_img, img, 10, 4) # 36 times
    # step 3
    img = correct_rotation(calibration_img, img)     #   1 time
    img = correct_offset(calibration_img, img, 5, 1) # 121 times

    # Change calibration image for the first frame (hotfix?)
    # if i == 0:
    calibration_img = img.copy()
    calibration_img = cv2.bitwise_and(calibration_img, calibration_img, mask=MASK)

    # Apply enhancement
    if ENHANCEMENT_NAME in ENHANCEMENTS.keys():
        img = enhance_image(img, ENHANCEMENTS[ENHANCEMENT_NAME])

    # Append image to the list
    output_frames.append(img)

    # Show processed image (only uncomment if you installed "opencv-python", because "opencv-python-headless" doesnt support guis)
    if SHOW_IMG_REALTIME == True:
        cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    # Print progress
    print("   done")

# Write output
print(f"INFO   saving out", end="", flush=True)
imageio.mimsave(OUTPUT_FILE, output_frames, loop=0, duration=MS_PER_FRAME)
print("   done")
