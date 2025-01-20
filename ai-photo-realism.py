import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import piexif
import os
from skimage.util import random_noise
from skimage.filters import gaussian
from datetime import datetime

def add_sensor_noise(image):
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 5, image.shape).astype(np.float32)
    # Add Salt and Pepper noise
    s_vs_p = 0.5
    amount = 0.004
    noisy_image = image.copy().astype(np.float32)
    # Salt noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255
    # Pepper noise
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0
    # Combine noises
    noisy_image += gaussian_noise
    # Clip to valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_motion_blur(image):
    # Randomly choose horizontal or vertical motion
    size = random.randint(3, 15)
    angle = random.choice([0, 90, random.randint(0, 180)])
    kernel = np.zeros((size, size))
    # Calculate the center point
    center = size // 2
    # Draw a line across the kernel
    cv2.line(kernel, (center, 0), (center, size-1), 1, thickness=1)
    # Rotate the kernel to the desired angle
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= np.sum(kernel)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_gaussian_blur(image):
    # Apply Gaussian blur with random sigma
    sigma = random.uniform(0.5, 2.5)
    blurred = gaussian(image, sigma=sigma, multichannel=True)
    blurred = (blurred * 255).astype(np.uint8)
    return blurred

def apply_lens_distortion(image):
    h, w = image.shape[:2]
    # Randomly choose distortion coefficients
    k1 = random.uniform(-0.3, 0.3)
    k2 = random.uniform(-0.2, 0.2)
    p1 = random.uniform(-0.001, 0.001)
    p2 = random.uniform(-0.001, 0.001)
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, p1, p2, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return distorted

def add_random_vignetting(image):
    rows, cols = image.shape[:2]
    # Randomly choose the intensity and spread of the vignetting
    intensity = random.uniform(0.5, 1.0)
    spread_x = random.uniform(300, 500)
    spread_y = random.uniform(300, 500)
    X_resultant_kernel = cv2.getGaussianKernel(cols, spread_x)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, spread_y)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    vignetting = image.astype(np.float32) * mask * intensity
    vignetting = np.clip(vignetting, 0, 255).astype(np.uint8)
    return vignetting

def color_jitter(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Random color adjustments
    color_factor = random.uniform(0.9, 1.1)
    brightness_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    hue_factor = random.uniform(-0.05, 0.05)
    saturation_factor = random.uniform(0.9, 1.1)
    
    img = ImageEnhance.Color(img).enhance(color_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    
    # Convert to HSV to adjust hue and saturation
    img = img.convert('HSV')
    np_img = np.array(img, dtype=np.float32)
    np_img[:, :, 0] = (np_img[:, :, 0] + hue_factor * 255) % 255
    np_img[:, :, 1] = np.clip(np_img[:, :, 1] * saturation_factor, 0, 255)
    img = Image.fromarray(np_img.astype(np.uint8), 'HSV')
    
    img = img.convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_and_interpolate(image):
    methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    num_resizes = random.randint(2, 4)  # Multiple resizing steps
    for _ in range(num_resizes):
        new_size = (random.randint(900, 1100), random.randint(900, 1100))
        method = random.choice(methods)
        image = cv2.resize(image, new_size, interpolation=method)
    return image

def embed_realistic_exif(image_path, output_path):
    exif_data = {
        "0th": {
            piexif.ImageIFD.Make: random.choice(["Canon", "Nikon", "Sony", "Fujifilm", "Olympus"]),
            piexif.ImageIFD.Model: random.choice(["EOS 5D Mark IV", "D850", "A7R IV", "X-T4", "E-M1 Mark III"]),
            piexif.ImageIFD.Software: "Adobe Photoshop Lightroom Classic 10.4 (Windows)",
            piexif.ImageIFD.DateTime: datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            piexif.ImageIFD.Orientation: random.choice([1, 3, 6, 8]),
            piexif.ImageIFD.XResolution: (300, 1),
            piexif.ImageIFD.YResolution: (300, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        },
        "Exif": {
            piexif.ExifIFD.LensMake: "Canon",
            piexif.ExifIFD.LensModel: random.choice(["EF50mm f/1.8 STM", "24-70mm f/2.8", "EF85mm f/1.4L IS USM"]),
            piexif.ExifIFD.FNumber: (random.randint(18, 28), 10),  # f/1.8 to f/2.8
            piexif.ExifIFD.ExposureTime: (1, random.choice([100, 200, 250, 500])),
            piexif.ExifIFD.ISOSpeedRatings: random.choice([100, 200, 400, 800, 1600]),
            piexif.ExifIFD.DateTimeOriginal: datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            piexif.ExifIFD.DateTimeDigitized: datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            piexif.ExifIFD.ShutterSpeedValue: (int(np.log2(random.randint(100, 1000))), 1),
            piexif.ExifIFD.ApertureValue: (random.randint(18, 28), 10),
            piexif.ExifIFD.BrightnessValue: (random.randint(-200, 200), 100),
            piexif.ExifIFD.ExposureBiasValue: (0, 1),
            piexif.ExifIFD.MaxApertureValue: (random.randint(14, 28), 10),
            piexif.ExifIFD.SubjectDistance: (1000, 10),
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: 'N',
            piexif.GPSIFD.GPSLatitude: ((40, 1), (0, 1), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: 'E',
            piexif.GPSIFD.GPSLongitude: ((74, 1), (0, 1), (0, 1)),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (10, 1),
        },
    }
    exif_bytes = piexif.dump(exif_data)
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # Ensure correct orientation
    img.save(output_path, "jpeg", exif=exif_bytes)

def scramble_pixels(image):
    h, w, _ = image.shape
    block_size = random.choice([4, 8, 16])
    for x in range(0, w, block_size):
        for y in range(0, h, block_size):
            sub_block = image[y:y+block_size, x:x+block_size]
            if sub_block.shape[0] != block_size or sub_block.shape[1] != block_size:
                continue  # Skip incomplete blocks
            # Shuffle pixels within the block
            flat = sub_block.reshape(-1, 3)
            np.random.shuffle(flat)
            image[y:y+block_size, x:x+block_size] = flat.reshape(sub_block.shape)
    return image

def add_hidden_watermark(image):
    watermark = np.zeros_like(image, dtype=np.float32)
    font_scale = random.uniform(0.5, 1.5)
    thickness = random.randint(1, 3)
    text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
    position = (random.randint(10, image.shape[1]//2), random.randint(10, image.shape[0]//2))
    cv2.putText(watermark, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    # Apply Gaussian blur to the watermark to make it less detectable
    watermark = cv2.GaussianBlur(watermark, (5, 5), 0)
    # Blend the watermark with the image
    watermarked = cv2.addWeighted(image.astype(np.float32), 1, watermark, random.uniform(0.0005, 0.002), 0)
    return watermarked.astype(np.uint8)

def metadata_cleaning(image_path):
    # Remove all existing metadata
    img = Image.open(image_path)
    data = list(img.getdata())
    img_without_exif = Image.new(img.mode, img.size)
    img_without_exif.putdata(data)
    temp_path = "clean_temp.jpg"
    img_without_exif.save(temp_path, "JPEG")
    return temp_path

def embed_additional_exif(output_path):
    # Additional EXIF fields for realism
    try:
        exif_dict = piexif.load(output_path)
    except:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
    # Add more EXIF data
    exif_dict["0th"][piexif.ImageIFD.XPTitle] = "Sample Image".encode('utf-16le')
    exif_dict["0th"][piexif.ImageIFD.XPComment] = "Processed by OpenAI Script".encode('utf-16le')
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = "Generated for educational purposes.".encode('utf-8')
    exif_bytes = piexif.dump(exif_dict)
    img = Image.open(output_path)
    img.save(output_path, "jpeg", exif=exif_bytes)

def simulate_jpeg_compression(image_path, quality=None):
    if quality is None:
        quality = random.choice([75, 85, 95])
    img = Image.open(image_path)
    img.save(image_path, "JPEG", quality=quality, optimize=True)
    return image_path

def process_image(image_path, output_path):
    # Step 1: Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Step 2: Metadata Cleaning
    temp_clean = metadata_cleaning(image_path)
    image = cv2.imread(temp_clean)
    os.remove(temp_clean)
    
    # Step 3: Apply Processing Steps in Random Order
    processing_steps = [
        add_sensor_noise,
        apply_motion_blur,
        apply_gaussian_blur,
        apply_lens_distortion,
        add_random_vignetting,
        color_jitter,
        resize_and_interpolate,
        scramble_pixels,
        add_hidden_watermark
    ]
    random.shuffle(processing_steps)
    for step in processing_steps:
        image = step(image)
    
    # Step 4: Save Intermediate Image
    temp_output = "temp_image.jpg"
    cv2.imwrite(temp_output, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    # Step 5: Embed Realistic EXIF Data
    embed_realistic_exif(temp_output, output_path)
    
    # Step 6: Embed Additional EXIF Data
    embed_additional_exif(output_path)
    
    # Step 7: Simulate JPEG Compression
    simulate_jpeg_compression(output_path)
    
    # Step 8: Final Check for Image Integrity
    final_image = cv2.imread(output_path)
    if final_image is None:
        print(f"Error: Unable to save processed image at {output_path}")
        return
    
    # Cleanup Temporary Files
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    print(f"Processed image saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    input_image = "ai_generated_image.jpg"
    output_image = "realistic_human_photo.jpg"
    process_image(input_image, output_image)
