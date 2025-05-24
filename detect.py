import cv2
import time
import os
from PIL import Image
from datetime import datetime
import argparse
from overflow_detector import OverflowDetector

def perpetual_webcam_detection(output_dir=None, 
                             interval_milliseconds=1000, 
                             save_images=False,
                             model_path='models/overflow_detector_v1_with_backbone.pt'):
    """
    Perpetually capture images from webcam, resize them using utils.utils.resize,
    and run MobileNetV3 inference using the OverflowDetector class.
    
    Args:
        output_dir: Directory to save captured images (None if not saving)
        interval_milliseconds: Interval in milliseconds between captures
        save_images: Whether to save captured images to disk
        model_path: Path to the trained model weights
    """
    # Initialize overflow detector
    print("🔧 Initializing Overflow Detector...")
    detector = OverflowDetector(model_path)
    
    # Create output directory if saving images
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Images will be saved to: {os.path.abspath(output_dir)}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    interval_seconds = interval_milliseconds / 1000.0
    print(f"✓ Webcam connected successfully!")
    print(f"✓ Running detection every {interval_milliseconds}ms ({interval_seconds}s)")
    print("✓ Press Ctrl+C to stop")
    print("=" * 80)
    
    capture_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(frame_rgb)
            
            # Run inference (OverflowDetector handles resizing internally)
            predicted_class, confidence, all_probs = detector.predict(pil_img)
            
            # Print results with enhanced formatting
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Frame {capture_count + 1:04d}:")
            print(f"  🔍 Prediction: {predicted_class.upper()} (confidence: {confidence:.3f})")
            
            # Color-coded probability display
            prob_display = []
            for i, class_name in enumerate(detector.classes):
                prob = all_probs[i]
                if class_name == predicted_class:
                    prob_display.append(f"✓ {class_name}: {prob:.3f}")
                else:
                    prob_display.append(f"  {class_name}: {prob:.3f}")
            
            print(f"  📊 Probabilities:")
            for prob_str in prob_display:
                print(f"    {prob_str}")
            
            # Alert for unsafe conditions
            if predicted_class == 'unsafe' and confidence > 0.7:
                print(f"  ⚠️  HIGH CONFIDENCE UNSAFE DETECTION! ⚠️")
            elif predicted_class == 'unsafe':
                print(f"  ⚠️  Unsafe condition detected")
            elif predicted_class == 'safe':
                print(f"  ✅ Safe condition")
            else:  # off
                print(f"  ⭕ System off")
            
            # Save image if requested
            if save_images and output_dir:
                timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                confidence_str = f"{confidence:.3f}".replace('.', '_')
                filename = os.path.join(
                    output_dir, 
                    f"detection_{timestamp_file}_{capture_count:04d}_{predicted_class}_{confidence_str}.jpg"
                )
                
                try:
                    pil_img.save(filename, quality=95)
                    print(f"  💾 Saved: {os.path.basename(filename)}")
                except Exception as e:
                    print(f"  ❌ Error saving image: {e}")
            
            print("-" * 80)
            capture_count += 1
            
            # Wait for the specified interval
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Detection stopped by user")
        print(f"📈 Total frames processed: {capture_count}")
        
        # Print session summary
        print("\n📊 Session Summary:")
        print(f"   • Total frames: {capture_count}")
        if capture_count > 0:
            avg_fps = capture_count / (capture_count * interval_seconds)
            print(f"   • Average FPS: {avg_fps:.2f}")
        if save_images and output_dir:
            print(f"   • Images saved to: {os.path.abspath(output_dir)}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📈 Frames processed before error: {capture_count}")
    
    finally:
        # Release the webcam
        cap.release()
        print("✓ Webcam released")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Overflow Detection System using MobileNetV3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--save', type=str, metavar='OUTPUT_DIR',
                        help='Save images to the specified directory')
    parser.add_argument('--interval', type=int, default=1000,
                        help='Interval in milliseconds between captures')
    parser.add_argument('--model_path', type=str, 
                        default='models/overflow_detector_v1_with_backbone.pt',
                        help='Path to the trained model weights')
    
    args = parser.parse_args()
    
    # Determine save settings
    save_images = args.save is not None
    output_dir = args.save if save_images else None
    
    print("🔍 Overflow Detection System")
    print("=" * 40)
    print(f"Model: MobileNetV3-Small")
    print(f"Classes: off, safe, unsafe")
    print(f"Mode: Webcam Detection")
    print(f"Interval: {args.interval}ms")
    print(f"Save images: {save_images}")
    if save_images:
        print(f"Output directory: {output_dir}")
    print("=" * 40)
    
    perpetual_webcam_detection(
        output_dir=output_dir,
        interval_milliseconds=args.interval,
        save_images=save_images,
        model_path=args.model_path
    )
