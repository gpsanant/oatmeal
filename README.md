# Video Frame Extraction Tool

This script extracts frames from videos at specified intervals, resizes them, and organizes them into directories.

## Requirements

Install the required packages:

```bash
pip install opencv-python pillow torchvision
```

## Usage

1. Place your video files in the `videos` directory (will be created if it doesn't exist)
2. Run the script:

```bash
python extract_frames.py
```

This will extract frames from all videos every 300ms and save them to the `snapshots` directory, with a separate subdirectory for each video.

## Command Line Arguments

- `--videos_dir`: Directory containing video files (default: 'videos')
- `--snapshots_dir`: Directory to save extracted frames (default: 'snapshots')
- `--interval_ms`: Interval in milliseconds between frames (default: 300)
- `--target_size`: Target size for resizing images (default: 224)

## Examples

Extract frames every 500ms:
```bash
python extract_frames.py --interval_ms 500
```

Use custom directories:
```bash
python extract_frames.py --videos_dir my_videos --snapshots_dir my_frames
```

Change the target size for resizing:
```bash
python extract_frames.py --target_size 512
``` 