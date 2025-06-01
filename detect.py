import cv2
import time
import os
from PIL import Image
from datetime import datetime
import argparse
from overflow_detector import OverflowDetector

def find_iphone_camera():
    """
    Find iPhone camera among available video devices.
    iPhone cameras typically appear as higher-numbered devices when connected.
    
    Returns:
        int: Camera index for iPhone, or 0 if not found
    """
    print("🔍 Searching for iPhone camera...")
    
    # Test camera indices (iPhone usually appears as index 1, 2, or 3)
    for camera_index in range(0, 10):
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Get camera name/info if possible
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
            # iPhone cameras typically have higher resolutions
            if width >= 1280 and height >= 720:
                print(f"✓ Found high-resolution camera at index {camera_index} ({int(width)}x{int(height)})")
                cap.release()
                return camera_index
            else:
                print(f"  Camera {camera_index}: {int(width)}x{int(height)} (likely built-in)")
            
            cap.release()
        else:
            break
    
    print("⚠️  iPhone camera not found, using default camera (index 0)")
    print("💡 Make sure your iPhone is connected and you've enabled camera sharing")
    return 0

def perpetual_webcam_detection(output_dir=None, 
                             interval_milliseconds=1000, 
                             save_images=False,
                             model_path='models/overflow_detector_v1_with_backbone.pt',
                             use_iphone=True):
    """
    Perpetually capture images from webcam (preferably iPhone), resize them using utils.utils.resize,
    and run MobileNetV3 inference using the OverflowDetector class.
    
    Args:
        output_dir: Directory to save captured images (None if not saving)
        interval_milliseconds: Interval in milliseconds between captures
        save_images: Whether to save captured images to disk
        model_path: Path to the trained model weights
        use_iphone: Whether to search for and use iPhone camera
    """
    # Initialize overflow detector
    print("🔧 Initializing Overflow Detector...")
    detector = OverflowDetector(model_path)
    
    # Create output directory if saving images
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Images will be saved to: {os.path.abspath(output_dir)}")
    
    # Find the best camera (iPhone if available)
    if use_iphone:
        camera_index = find_iphone_camera()
    else:
        camera_index = 0
        print("📷 Using default camera (index 0)")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera at index {camera_index}")
        if camera_index != 0:
            print("🔄 Trying default camera...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Error: Could not open any camera")
                return
            camera_index = 0
        else:
            return
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    interval_seconds = interval_milliseconds / 1000.0
    print(f"✓ Camera connected successfully!")
    print(f"✓ Using camera index: {camera_index}")
    print(f"✓ Resolution: {actual_width}x{actual_height}")
    print(f"✓ Running detection every {interval_milliseconds}ms ({interval_seconds}s)")
    print("✓ Press Ctrl+C to stop")
    print("=" * 80)
    
    capture_count = 0
    
    # Timing statistics tracking
    total_preprocess_time = 0
    total_inference_time = 0
    total_prediction_time = 0
    
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
            predicted_class, confidence, all_probs, timing_info = detector.predict(pil_img)
            
            # Print results with enhanced formatting
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Frame {capture_count + 1:04d}:")
            print(f"  🔍 Prediction: {predicted_class.upper()} (confidence: {confidence:.3f})")
            
            # Display timing information
            print(f"  ⏱️  Timing:")
            print(f"    • Preprocessing: {timing_info['preprocess_time']*1000:.1f}ms")
            print(f"    • Model inference: {timing_info['inference_time']*1000:.1f}ms")
            print(f"    • Total time: {timing_info['total_time']*1000:.1f}ms")
            print(f"    • Inference FPS: {timing_info['fps']:.1f}")
            
            # Accumulate timing statistics
            total_preprocess_time += timing_info['preprocess_time']
            total_inference_time += timing_info['inference_time']
            total_prediction_time += timing_info['total_time']
            
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
                filename = os.path.join(
                    output_dir, 
                    f"{capture_count}.jpg"
                )
                
                try:
                    pil_img.save(filename)
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
            
            # Timing statistics
            avg_preprocess = (total_preprocess_time / capture_count) * 1000
            avg_inference = (total_inference_time / capture_count) * 1000
            avg_total = (total_prediction_time / capture_count) * 1000
            avg_inference_fps = capture_count / total_prediction_time if total_prediction_time > 0 else 0
            
            print(f"   • Average timing per frame:")
            print(f"     - Preprocessing: {avg_preprocess:.1f}ms")
            print(f"     - Model inference: {avg_inference:.1f}ms")
            print(f"     - Total prediction: {avg_total:.1f}ms")
            print(f"     - Inference FPS: {avg_inference_fps:.1f}")
            
        if save_images and output_dir:
            print(f"   • Images saved to: {os.path.abspath(output_dir)}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"📈 Frames processed before error: {capture_count}")
        
        # Print timing statistics even on error
        if capture_count > 0:
            avg_preprocess = (total_preprocess_time / capture_count) * 1000
            avg_inference = (total_inference_time / capture_count) * 1000
            avg_total = (total_prediction_time / capture_count) * 1000
            avg_inference_fps = capture_count / total_prediction_time if total_prediction_time > 0 else 0
            
            print(f"📊 Timing Statistics (before error):")
            print(f"   • Average preprocessing: {avg_preprocess:.1f}ms")
            print(f"   • Average inference: {avg_inference:.1f}ms")
            print(f"   • Average total: {avg_total:.1f}ms")
            print(f"   • Average inference FPS: {avg_inference_fps:.1f}")
    
    finally:
        # Release the webcam
        cap.release()
        print("✓ Camera released")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Overflow Detection System using MobileNetV3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--save', action='store_true',
                        help='Save images to a timestamped directory under /detect/')
    parser.add_argument('--interval', type=int, default=300,
                        help='Interval in milliseconds between captures')
    parser.add_argument('--model_path', type=str, 
                        default='models/overflow_detector_v1_with_backbone.pt',
                        help='Path to the trained model weights')
    parser.add_argument('--no_iphone', action='store_true',
                        help='Use default camera instead of searching for iPhone')
    
    args = parser.parse_args()
    
    # Determine save settings
    save_images = args.save
    output_dir = None
    if save_images:
        # Create timestamped directory under detect/
        timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"detect/{timestamp_folder}"
    
    use_iphone = not args.no_iphone
    
    print("🔍 Overflow Detection System")
    print("=" * 40)
    print(f"Model: MobileNetV3-Small")
    print(f"Classes: off, safe, unsafe")
    print(f"Mode: {'iPhone' if use_iphone else 'Default'} Camera Detection")
    print(f"Interval: {args.interval}ms")
    print(f"Save images: {save_images}")
    if save_images:
        print(f"Output directory: {output_dir}")
    print("=" * 40)
    
    perpetual_webcam_detection(
        output_dir=output_dir,
        interval_milliseconds=args.interval,
        save_images=save_images,
        model_path=args.model_path,
        use_iphone=use_iphone
    )
