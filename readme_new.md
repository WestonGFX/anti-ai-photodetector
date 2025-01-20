# AI Image Realism Enhancer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Processing Steps](#processing-steps)
- [Dependencies](#dependencies)
- [Customization](#customization)
- [Ethical Considerations](#ethical-considerations)
- [License](#license)

## Overview

**AI Image Realism Enhancer** is a powerful Python script designed to transform AI-generated images into highly realistic photographs. By applying a series of sophisticated image processing techniques and embedding authentic EXIF metadata, the tool ensures that the resulting images are indistinguishable from genuine human-taken photos to both the human eye and AI detection algorithms.

> **⚠️ Disclaimer:** This tool is intended solely for educational and research purposes. Misuse of this software may have ethical and legal implications. Always adhere to applicable laws and ethical guidelines when using or distributing manipulated images.

## Features

- **Diverse Noise Addition:** Simulates various sensor imperfections including Gaussian and salt-and-pepper noise.
- **Advanced Blur Techniques:** Implements motion blur and Gaussian blur to mimic camera movements and lens softness.
- **Enhanced Lens Distortion:** Applies sophisticated distortion models to emulate different lens characteristics.
- **Dynamic Vignetting:** Creates natural-looking light fall-off towards image edges with adjustable intensity and spread.
- **Comprehensive Color Adjustments:** Modifies hue, saturation, brightness, and contrast to replicate camera color profiles and lighting conditions.
- **Multiple Resizing and Interpolation:** Introduces subtle artifacts through multiple resizing steps with varying interpolation methods.
- **Pixel Shuffling with Variable Block Sizes:** Randomizes pixel blocks to prevent pattern detection and enhance unpredictability.
- **Steganographic Watermarking:** Embeds semi-transparent, blurred text watermarks to include unique identifiers discreetly.
- **Realistic EXIF Data Embedding:** Populates a wide range of EXIF fields with plausible values, including GPS data, camera model, lens information, and software details.
- **Metadata Cleaning:** Removes existing metadata to eliminate traces of AI generation.
- **JPEG Compression Simulation:** Applies varying compression levels to mimic different camera qualities and reduce file size.
- **Randomized Processing Order:** Shuffles processing steps to avoid predictable patterns detectable by AI algorithms.

## Demo

*Include screenshots or example images demonstrating the transformation from AI-generated to realistic photos.*

![Before and After](assets/before_after_example.jpg)

## Installation

### Prerequisites

- **Python 3.7+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/ai-image-realism-enhancer.git
cd ai-image-realism-enhancer
```

Install Dependencies
It's recommended to use a virtual environment to manage dependencies.

bash
Copy
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
If a requirements.txt is not provided, you can install the necessary packages individually:

bash
Copy
pip install opencv-python pillow piexif scikit-image numpy
Usage
Basic Usage
bash
Copy
python process_image.py --input path/to/ai_generated_image.jpg --output path/to/realistic_photo.jpg
Script Parameters
--input: Path to the AI-generated input image.
--output: Path where the processed realistic image will be saved.
Example
bash
Copy
python process_image.py --input samples/ai_generated_image.jpg --output outputs/realistic_photo.jpg
Running the Script
Ensure that you have the necessary permissions to read the input file and write to the output directory.

bash
Copy
python process_image.py --input ai_generated_image.jpg --output realistic_human_photo.jpg
Upon successful execution, the processed image will be saved at the specified output path with realistic EXIF metadata and enhanced visual attributes.

Processing Steps
The script applies the following steps to transform the AI-generated image:

Metadata Cleaning: Strips all existing metadata to remove traces of AI generation.
Sensor Noise Addition: Adds Gaussian and salt-and-pepper noise to simulate sensor imperfections.
Motion and Gaussian Blur: Applies randomized motion blur (direction and intensity) and Gaussian blur for lens softness.
Lens Distortion: Implements random distortion coefficients to emulate various lens imperfections.
Vignetting: Applies dynamic vignetting with random intensity and spread to create natural light fall-off.
Color Jitter: Adjusts hue, saturation, brightness, and contrast to mimic camera color profiles.
Resizing and Interpolation: Performs multiple resizing steps with different interpolation methods to introduce subtle artifacts.
Pixel Shuffling: Randomizes pixels within variable-sized blocks to prevent pattern detection.
Hidden Watermarking: Embeds a semi-transparent, blurred text watermark with unique identifiers.
EXIF Data Embedding: Populates comprehensive EXIF fields with realistic values, including camera details and GPS data.
JPEG Compression Simulation: Applies varying JPEG compression levels to replicate different camera qualities.
Randomized Processing Order: Shuffles the sequence of processing steps to avoid predictable patterns.
Dependencies
The script relies on the following Python libraries:

OpenCV (opencv-python)
Pillow (Pillow)
piexif (piexif)
scikit-image (scikit-image)
NumPy (numpy)
All dependencies can be installed via pip as shown in the Installation section.

Customization
You can customize various aspects of the processing steps to better suit your specific needs:

Noise Levels: Adjust the mean and standard deviation in the add_sensor_noise function.
Blur Intensity: Modify the range of kernel sizes and sigma values in blur functions.
Distortion Coefficients: Change the ranges for distortion coefficients in apply_lens_distortion.
Vignetting Parameters: Tweak intensity and spread ranges in add_random_vignetting.
Color Adjustment Factors: Customize the ranges for hue, saturation, brightness, and contrast in color_jitter.
Resizing Steps: Alter the number of resizing iterations and size ranges in resize_and_interpolate.
Watermark Text: Modify the watermark text generation logic in add_hidden_watermark.
Feel free to explore and adjust these parameters to achieve the desired level of realism and imperceptibility.

Ethical Considerations
Manipulating images to disguise their origin can have significant ethical and legal implications, including:

Misinformation: Altered images can be used to deceive or spread false information.
Intellectual Property: Unauthorized modification and distribution of images may infringe on copyrights.
Privacy Concerns: Embedding GPS data or other metadata can inadvertently expose sensitive information.
Use this tool responsibly and ensure compliance with all relevant laws and ethical standards. Always obtain necessary permissions when modifying and sharing images that are not your own.

License
This project is licensed under the MIT License.
