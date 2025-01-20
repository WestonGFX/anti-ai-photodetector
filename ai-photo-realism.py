import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import piexif
import os
from skimage.util import random_noise
from skimage.filters import gaussian

def add_sensor_noise(image):
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def apply_motion_blur(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_lens_distortion(image):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w//2], [0, h, h//2], [0, 0, 1]], dtype=np.float32)
    D = np.array([0.05, -0.02, 0.001, 0.0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), 5)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

def add_random_vignetting(image):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, random.uniform(200, 300))
    Y_resultant_kernel = cv2.getGaussianKernel(rows, random.uniform(200, 300))
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    return cv2.addWeighted(image, 1, np.stack([mask]*3, axis=-1).astype(np.uint8), -0.3, 0)

def color_jitter(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.95, 1.05))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.95, 1.05))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.95, 1.05))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_and_interpolate(image):
    methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    return cv2.resize(image, (random.randint(950, 1050), random.randint(950, 1050)), interpolation=random.choice(methods))

def embed_realistic_exif(image_path, output_path):
    exif_data = {
        "0th": {
            piexif.ImageIFD.Make: random.choice(["Canon", "Nikon", "Sony"]),
            piexif.ImageIFD.Model: random.choice(["EOS 5D Mark IV", "D850", "A7R IV"]),
            piexif.ImageIFD.DateTime: "2024:01:19 14:33:00",
        },
        "Exif": {
            piexif.ExifIFD.LensMake: "Canon",
            piexif.ExifIFD.LensModel: random.choice(["EF50mm f/1.8 STM", "24-70mm f/2.8"]),
            piexif.ExifIFD.FNumber: (50, 10),
            piexif.ExifIFD.ExposureTime: (1, 200),
            piexif.ExifIFD.ISOSpeedRatings: 400,
        },
    }
    exif_bytes = piexif.dump(exif_data)
    img = Image.open(image_path)
    img.save(output_path, "jpeg", exif=exif_bytes)

def scramble_pixels(image):
    h, w, _ = image.shape
    block_size = 8
    for x in range(0, w, block_size):
        for y in range(0, h, block_size):
            sub_block = image[y:y+block_size, x:x+block_size]
            np.random.shuffle(sub_block)
            image[y:y+block_size, x:x+block_size] = sub_block
    return image

def add_hidden_watermark(image):
    watermark = np.zeros_like(image)
    cv2.putText(watermark, "12345", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return cv2.addWeighted(image, 1, watermark, 0.001, 0)

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = add_sensor_noise(image)
    image = apply_motion_blur(image)
    image = apply_lens_distortion(image)
    image = add_random_vignetting(image)
    image = color_jitter(image)
    image = resize_and_interpolate(image)
    image = scramble_pixels(image)
    image = add_hidden_watermark(image)

    temp_output = "temp_image.jpg"
    cv2.imwrite(temp_output, image)
    embed_realistic_exif(temp_output, output_path)
    os.remove(temp_output)

    print(f"Processed image saved at: {output_path}")

# Example usage
process_image("ai_generated_image.jpg", "realistic_human_photo.jpg")
