---

# AI Image Humanization Script

## Overview
This project provides a Python script designed to process AI-generated images and make them appear indistinguishable from real human-taken photographs. The script applies various techniques such as:

- **Sensor Noise Addition:** Simulates imperfections found in real camera sensors.
- **Motion Blur Simulation:** Introduces slight blurring to mimic real-world movement.
- **Lens Distortion:** Applies natural optical distortions.
- **Vignetting Effects:** Adds subtle darkening around the edges.
- **Color Adjustments:** Introduces slight variations in color, brightness, and contrast.
- **Pixel Scrambling:** Reorders small sections of the image to break AI detection patterns.
- **Hidden Watermarking:** Embeds low-opacity patterns to simulate real photo artifacts.
- **EXIF Metadata Injection:** Adds realistic camera data such as model, lens, and exposure.

## Installation
Ensure you have Python 3 installed, then install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
To process an AI-generated image and make it look more human-like, run the script with:

```bash
python process_image.py
```

### Example
1. Place your AI-generated image in the project directory and rename it to `ai_generated_image.jpg`.
2. Run the script and check the output in `realistic_human_photo.jpg`.

## Configuration
You can adjust various parameters in the script, such as:

- `kernel_size` for motion blur
- `resize_and_interpolate` method choices
- `random noise intensity`

Modify these values in the `process_image` function to fine-tune the results.

## Dependencies
Refer to `requirements.txt` for the full list of required packages.

## Ethical Considerations
This script is for **research and educational purposes only.** Any misuse of the tool for deceptive or fraudulent activities is strongly discouraged.

## License
This project is licensed under the MIT License.

---

