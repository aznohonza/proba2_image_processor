# SWAP Solar Image Processor

A Python tool for processing and compiling SWAP solar observatory images into animated GIFs or WebP animations.

## Overview

This tool processes raw SWAP (Sun Watcher using Active Pixel System detector and Image Processing) images from the PROBA2 satellite and combines them into smooth animations. The script performs several image correction operations:

- Automatic rotation correction to align images
- Image enhancement with various color profiles
- Blurry and corrupted frame detection and removal
- Compilation into animated GIF/WebP formats

## Example Output

![SWAP Solar Animation Example](https://github.com/aznohonza/proba2_image_processor/raw/main/examples/example_output.gif)

*Example animation of solar activity captured by PROBA2's SWAP instrument and processed with this tool.*

## Requirements

- Python 3.6+ (tested on Python 3.12.9 but should be okay at lower versions too)
- Dependencies:
  - numpy
  - imageio
  - opencv-python-headless (or opencv-python if you want realtime display of processed frames)
  - os
  - re

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:

```bash
pip install numpy imageio opencv-python-headless
```

Note: If you want realtime image display, you can use the normal version:

```bash
pip install numpy imageio opencv-python
```

## Usage

1. Place your SWAP images in a folder (e.g., "SWAP/")
2. Make sure you have the calibration image (`.calibration.png`) in the same directory as the script
3. Configure the settings at the top of the script
4. Run the script:

```bash
python main.py
```

## Configuration Options

Edit these variables at the top of the script to customize processing:

- `INPUT_DIR` - Directory containing SWAP images
- `OUTPUT_FILE` - Output filename (supports .gif or .webp)
- `MS_PER_FRAME` - Milliseconds per frame in the output animation
- `EXTRA_PROCESSING` - Enable additional alignment processing (rarely needed)
- `ENHANCEMENT_NAME` - Color profile to apply ("official", "blue", "orange", etc. or None)
- `SHOW_IMG_REALTIME` - Display frames during processing (requires opencv-python)

## Color Profiles

Several enhancement profiles are available:

- `official` - Standard SWAP coloring (orange)
- `official_blue` - Blue variant of the official coloring
- `orange` - High contrast orange
- `blue` - High contrast blue
- `green` - High contrast green
- `pink` - High contrast pink
- `magenta` - High contrast magenta
- `purple` - High contrast purple

## Processing Pipeline

The script:

1. Loads and sorts images based on their timestamp
2. Calculates image quality using the Tenengrad measure
3. Skips blurry or corrupted frames
4. Performs multi-step image alignment:
   - Rotation correction
   - Coarse alignment
   - Medium alignment
   - Fine alignment
5. Applies color enhancement
6. Compiles frames into an animation

## Notes

- First loaded image becomes the reference for alignment
- A circular mask is applied to focus on the sun disk and reduce computation

## Example

```bash
# Set to process images in the SWAP folder
INPUT_DIR = "SWAP/"

# Output as GIF with 100ms per frame
OUTPUT_FILE = "output.gif"
MS_PER_FRAME = 100

# Use the official SWAP coloring
ENHANCEMENT_NAME = "official"
```

## Acknowledgements

- Special thanks to @soplica.pl on Discord for providing the calibration image and test data.
- Special thanks to @seler1500 on Discord for providing test data.
