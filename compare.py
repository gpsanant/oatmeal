from PIL import Image
from overflow_detector import OverflowDetector
from utils.utils import resize
import torch
import os

def debug_preprocessing(image_path, model_path='models/overflow_detector_v1_with_backbone.pt'):
    """
    Debug the preprocessing differences between predict(img) vs predict(resize(img))
    """
    detector = OverflowDetector(model_path)
    
    # Load original image
    img = Image.open(image_path)
    print(f"Original image size: {img.size}")
    # Manual resize
    img = resize(img)
    img.save(f"out/resized.jpg")
    print(f"After manual resize: {img.size}")

    img_load = Image.open(f"out/resized.jpg")
    print(f"After loading resized image: {img_load.size}")
    
    # Test 1: predict(original_img) - this calls resize() internally
    print("\n=== Test 1: predict(original_img) ===")
    pred1, conf1, probs1, _ = detector.predict(img)
    print(f"Prediction: {pred1} (confidence: {conf1:.3f})")
    
    # Test 2: predict(resized_img) - resize already applied
    print("\n=== Test 2: predict(resize(img)) ===")
    pred2, conf2, probs2, _ = detector.predict(img_load)
    print(f"Prediction: {pred2} (confidence: {conf2:.3f})")
    
    # Check if they're different
    if pred1 != pred2:
        print(f"\n⚠️  DIFFERENT PREDICTIONS!")
        print(f"predict(img): {pred1} ({conf1:.3f})")
        print(f"predict(resize(img)): {pred2} ({conf2:.3f})")
        
        # Let's check the tensor shapes after preprocessing
        print("\n=== Debugging tensor shapes ===")
        
        # Manually trace the preprocessing for original image
        img1_after_resize = resize(img)
        print(f"Original -> resize(): {img1_after_resize.size}")
        tensor1 = detector.transform(img1_after_resize)
        print(f"After transform: {tensor1.shape}")
        
        # Manually trace the preprocessing for pre-resized image  
        img2_after_resize = resize(img_load)  # This is resize(resize(img))!
        print(f"Resized -> resize(): {img2_after_resize.size}")
        tensor2 = detector.transform(img2_after_resize)
        print(f"After transform: {tensor2.shape}")
        
        print(f"\n🔍 The issue: resize(resize(img)) != resize(img)")
        print(f"Double resize changes the image!")
        
    else:
        print(f"\n✅ Same predictions: {pred1} ({conf1:.3f})")

def predict_image(image_path, model_path='models/overflow_detector_v1_with_backbone.pt'):
    """
    Predict overflow status for a single image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model weights
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Initialize detector
    detector = OverflowDetector(model_path)
    
    # Load and predict
    image = Image.open(image_path)
    predicted_class, confidence, all_probs = detector.predict(image)
    
    return predicted_class, confidence, all_probs

def predict_multiple_images(image_paths, model_path='models/overflow_detector_v1_with_backbone.pt'):
    """
    Predict overflow status for multiple images.
    
    Args:
        image_paths: List of image file paths
        model_path: Path to the trained model weights
        
    Returns:
        list: List of (image_path, predicted_class, confidence, all_probabilities) tuples
    """
    # Initialize detector once
    detector = OverflowDetector(model_path)
    
    results = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            predicted_class, confidence, all_probs, timing_info = detector.predict(image)
            results.append((image_path, predicted_class, confidence, all_probs, timing_info))
        except Exception as e:
            print(f"❌ Error with {image_path}: {e}")
            results.append((image_path, None, None, None))
    
    return results

def predict_directory(directory_path, model_path='models/overflow_detector_v1_with_backbone.pt'):
    """
    Predict overflow status for all images in a directory.
    
    Args:
        directory_path: Path to directory containing images
        model_path: Path to the trained model weights
        
    Returns:
        list: List of (image_path, predicted_class, confidence, all_probabilities) tuples
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(directory_path, filename))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {directory_path}")
    return predict_multiple_images(image_paths, model_path)

# Example usage - edit these as needed
if __name__ == "__main__":
    
    # Debug the preprocessing issue
    # debug_preprocessing("prod/detection_20250524_172455_0081_off_0_449.jpg")
    
    print("\n" + "="*80 + "\n")
    
    # image_paths = [
    #     "detect/20250531_145223/96.jpg"
    # ]
    # results = predict_multiple_images(image_paths)
    # for path, pred, conf, probs, timing_info in results:
    #     if pred:
    #         print(f"{os.path.basename(path)}: {pred} ({conf:.3f})")
    #         print(f"All probabilities: {dict(zip(['off', 'safe', 'unsafe'], probs))}")
    #         print(f"Timing info: {timing_info}")
    
    # Directory prediction
    results = predict_directory("detect/20250531_153942")
    for path, pred, conf, probs, timing_info in results:
        if pred:
            print(f"{os.path.basename(path)}: {pred} ({conf:.3f})")
            print(f"Timing info: {timing_info}")
            # print(f"All probabilities: {dict(zip(['off', 'safe', 'unsafe'], probs))}")