import os
import cv2
from PIL import Image
import argparse
import shutil

def resize(img, target_size=224):
    # Resize keeping aspect ratio so longer side = target_size
    w, h = img.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    img = img.resize((new_w, new_h))
    return img

def extract_frames(video_path, output_dir, interval_ms=300, target_size=224):
    """
    Extract frames from a video at specified millisecond intervals and resize them.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval_ms: Interval in milliseconds between frames
        target_size: Target size for resizing images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video name for filename prefix
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * (interval_ms / 1000))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(frame_rgb)
            
            # Resize image
            pil_img = resize(pil_img, target_size)
            
            # Save the frame with video name in filename
            frame_filename = os.path.join(output_dir, f"{video_name}_{saved_count:05d}.jpg")
            pil_img.save(frame_filename)
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames from {os.path.basename(video_path)} to {output_dir}")
    return saved_count

def process_videos(videos_dir, snapshots_dir, interval_ms=300, target_size=224):
    """
    Process all videos in a directory and extract frames.
    
    Args:
        videos_dir: Directory containing video files
        snapshots_dir: Directory to save extracted frames
        interval_ms: Interval in milliseconds between frames
        target_size: Target size for resizing images
    """
    # Create videos and snapshots directories if they don't exist
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    
    # List all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, _, files in os.walk(videos_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    total_frames = 0
    
    # Process each video
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(snapshots_dir, video_name)
        
        # Clear existing output directory if it exists
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        
        frames = extract_frames(
            video_path, 
            video_output_dir, 
            interval_ms=interval_ms,
            target_size=target_size
        )
        total_frames += frames
    
    print(f"Processed {len(video_files)} videos, extracted {total_frames} frames in total")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos at specified intervals')
    parser.add_argument('--videos_dir', type=str, default='videos', 
                        help='Directory containing video files')
    parser.add_argument('--snapshots_dir', type=str, default='snapshots', 
                        help='Directory to save extracted frames')
    parser.add_argument('--interval_ms', type=int, default=300, 
                        help='Interval in milliseconds between frames')
    parser.add_argument('--target_size', type=int, default=224, 
                        help='Target size for resizing images')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths for clarity
    videos_dir = os.path.abspath(args.videos_dir)
    snapshots_dir = os.path.abspath(args.snapshots_dir)
    
    process_videos(
        videos_dir, 
        snapshots_dir, 
        interval_ms=args.interval_ms,
        target_size=args.target_size
    )

if __name__ == "__main__":
    main() 