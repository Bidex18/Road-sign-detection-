import cv2
import argparse
import numpy as np
import os

# Define sign categories and their IDs
sign_categories = {
    "warning": {
        "roundabout": 1,
        "double bend": 2,
        "dual carriageway ends": 3,
        "traffic lights": 4,
        "roadworks": 5,
        "ducks": 6,
        "unknown": 101
    },
    "imperative": {
        "turn left": 7,
        "keep left": 8,
        "mini roundabout": 9,
        "one way": 10,
        "unknown": 102
    },
    "prohibition": {
        "warning": 11,
        "give way": 12,
        "no entry": 13,
        "stop": 14,
        "unknown": 103
    },
    "speed limit": {
        "20MPH": 15,
        "30MPH": 16,
        "40MPH": 17,
        "50MPH": 18,
        "national speed limit": 19,
        "unknown": 104
    }
}

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    blur_threshold = 100
    return variance < blur_threshold

# Function to apply Wiener deblurring
def deblur_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Create a Gaussian blur kernel (estimate of the blur kernel)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  # Simulate a blur kernel

    # Apply Wiener filter
    epsilon = 1e-4  # Regularization parameter
    restored = np.conj(mask) / (np.abs(mask) ** 2 + epsilon) * f_shift
    restored = np.fft.ifftshift(restored)
    restored = np.fft.ifft2(restored)
    restored = np.abs(restored)

    # Normalize the restored image
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Merge back to BGR for consistency
    deblurred = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
    return deblurred    

def enhance_image(image):
    if detect_blur(image):
        image = deblur_image(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)
    return denoised

def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    sides = len(approx)
    
    if sides == 3:
        return "triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
    elif sides == 5:
        return "pentagon"
    elif sides == 6:
        return "hexagon"    
    elif sides == 8:
        return "octagon"
    elif sides > 8:
        return "circle"
    else:
        return "complex"

def identify_sign(color, shape,image, contour):
    
    if color == "red":
            if shape == "circle":
                
                # Check for white region within the red circle
                # Step 1: Mask the red circle
                red_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(red_mask, [contour], 0, 255, -1)
                white_region = cv2.bitwise_and(image, image, mask=red_mask)

                # Step 2: Check for white region within the red circle
                gray_white_region = cv2.cvtColor(white_region, cv2.COLOR_BGR2GRAY)
                # Step 3: Apply adaptive thresholding to detect the white region
                binary = cv2.adaptiveThreshold(
                    gray_white_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

                # Step 5: Count white pixels
                white_pixels = cv2.countNonZero(binary)
                
                if white_pixels > 0.2 * cv2.contourArea(contour):  # Adjust ratio if necessary
                    print("White region detected. Checking for black text.")
                    return {"name": "speed limit", "number": sign_categories["speed limit"]["unknown"]}
                    
                # If no significant white region, classify as "prohibition"
                return {"name": "prohibition", "number": sign_categories["prohibition"]["unknown"]}

            elif shape == "triangle":
                return {"name": "warning", "number": sign_categories["warning"]["unknown"]}

            elif shape == "octagon":
                return {"name": "stop", "number": sign_categories["prohibition"]["stop"]}

            
    elif color == "blue":
        if shape == "circle":
            return {"name": "mandatory", "number": sign_categories["imperative"]["unknown"]}
        elif shape == "rectangle":
            return {"name": "information", "number": sign_categories["imperative"]["unknown"]}
    elif color == "green" and shape == "rectangle":
        return {"name": "direction primary", "number": 0}
    elif color == "white":
        if shape == "rectangle":
            return {"name": "direction non-primary", "number": 0}
        elif shape == "circle":
            return {"name": "national speed limit", "number": sign_categories["speed limit"]["national speed limit"]}
    elif color == "yellow" and shape == "rectangle":
        return {"name": "temporary", "number": 0}
    elif color == "brown" and shape == "rectangle":
        return {"name": "tourist", "number": 0}
    elif color == "orange" and shape == "rectangle":
        return {"name": "temporary", "number": 0}
    return None

def detect_signs(image):
    enhanced = enhance_image(image)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Define color ranges for different road signs
    color_ranges = {
        "red": (np.array([0, 100, 100], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        "red2": (np.array([160, 100, 100], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        "blue": (np.array([100, 100, 100], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)),
        "yellow": (np.array([20, 100, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)),
        "white": (np.array([0, 0, 200], dtype=np.uint8), np.array([180, 30, 255], dtype=np.uint8))
    }

    detected_signs = []

    for color, (lower, upper) in color_ranges.items():
        if color == "red":
            mask = cv2.bitwise_or(cv2.inRange(hsv, color_ranges["red"][0], color_ranges["red"][1]),
                                  cv2.inRange(hsv, color_ranges["red2"][0], color_ranges["red2"][1]))
        else:
            mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue

            # Classify shape
            shape = classify_shape(contour)

            # Identify sign based on color and shape
            sign_info = identify_sign(color, shape, enhanced, contour)

            if sign_info:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = calculate_confidence(shape, color, enhanced, cv2.contourArea(contour))
                
                
                detected_signs.append({
                    "sign_name": sign_info["name"],
                    "sign_number": sign_info["number"],
                    "bb_xcentre": (x + w / 2) / image.shape[1],
                    "bb_ycentre": (y + h / 2) / image.shape[0],
                    "bb_width": w / image.shape[1],
                    "bb_height": h / image.shape[0],
                    "confidence": confidence
                })

    return detected_signs

def calculate_confidence(shape, color, image, area):
    base_confidence = 0.3
    
    shape_confidence = {"circle": 0.2, "triangle": 0.2, "octagon": 0.2, "rectangle": 0.15, "square": 0.15}
    base_confidence += shape_confidence.get(shape, 0.1)
    
    color_confidence = {"red": 0.15, "blue": 0.15, "yellow": 0.1}
    base_confidence += color_confidence.get(color, 0.05)
    
    if not detect_blur(image):
        base_confidence += 0.1
    
    area_confidence = min(area / 10000, 0.2)  
    base_confidence += area_confidence
     
    return min(base_confidence, 1.0)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return []
    detections = detect_signs(image)
    for detection in detections:
        detection["filename"] = image_path
        detection["frame_number"] = 0
        detection["timestamp"] = 0
    return detections

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 480))
        frame_detections = detect_signs(resized_frame)
        
        for detection in frame_detections:
            detection["filename"] = video_path
            detection["frame_number"] = frame_number
            detection["timestamp"] = frame_number / cap.get(cv2.CAP_PROP_FPS)
            detections.append(detection)
        
        frame_number += 1
    
    cap.release()
    return detections

def process_multiple_images(image_list_path):
    detections = []
    with open(image_list_path, 'r') as f:
        for line in f:
            image_path = line.strip()
            if not image_path:
                continue
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist.")
                continue
            detections.extend(process_image(image_path))
    return detections

def write_output(detections, output_file):
    # Group detections by filename
    from collections import defaultdict
    grouped_detections = defaultdict(list)
    for detection in detections:
        grouped_detections[detection['filename']].append(detection)

    with open(output_file, 'w') as f:
        # Process each image
        for filename, image_detections in grouped_detections.items():
            # Filter detections by confidence threshold
            confidence_threshold = 0.5
            filtered_detections = [d for d in image_detections if d['confidence'] >= confidence_threshold]

            if filtered_detections:
                # Select the detection with the highest confidence
                best_detection = max(filtered_detections, key=lambda x: x['confidence'])

                # Validate bounding box coordinates
                best_detection['bb_xcentre'] = min(max(best_detection['bb_xcentre'], 0), 1)
                best_detection['bb_ycentre'] = min(max(best_detection['bb_ycentre'], 0), 1)
                best_detection['bb_width'] = min(max(best_detection['bb_width'], 0), 1)
                best_detection['bb_height'] = min(max(best_detection['bb_height'], 0), 1)

                # Write the best detection to the output file
                line = (
                    f"{best_detection['filename']},"
                    f"{best_detection['sign_number']},"
                    f"{best_detection['sign_name']},"
                    f"{best_detection['bb_xcentre']:.4f},"
                    f"{best_detection['bb_ycentre']:.4f},"
                    f"{best_detection['bb_width']:.4f},"
                    f"{best_detection['bb_height']:.4f},"
                    f"{best_detection['frame_number']},"
                    f"{best_detection['timestamp']:.1f},"
                    f"{best_detection['confidence']:.1f}\n"
                )
                f.write(line)
def main():
    parser = argparse.ArgumentParser(description='Road Sign Detection')
    parser.add_argument('--image', help='Path to a single image file')
    parser.add_argument('--inputfile', help='Path to a text file containing multiple image paths')
    parser.add_argument('--video', help='Path to a video file')
    parser.add_argument('--output', default='output.txt', help='Path to output file')
    parser.add_argument('--interactive', action='store_true', help='Display results interactively')
    
    args = parser.parse_args()
    
       
    if args.image:
        detections = process_image(args.image)
    elif args.inputfile:
        detections = process_multiple_images(args.inputfile)
    elif args.video:
        detections = process_video(args.video)
    else:
        
        return
    
    write_output(detections, args.output)
    
    if args.interactive:
        
        pass

if __name__ == "__main__":
    main()
