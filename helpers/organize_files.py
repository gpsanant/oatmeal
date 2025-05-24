import os
import shutil

# Source and destination directories
source_dirs = ['data2/off', 'data2/safe', 'data2/unsafe']
dest_base_dir = 'data3'

# Create the base destination directory if it doesn't exist
if not os.path.exists(dest_base_dir):
    os.makedirs(dest_base_dir)

# Process each category directory
for source_dir in source_dirs:
    category = os.path.basename(source_dir)  # 'off', 'safe', or 'unsafe'
    
    # Create the category directory in data3 if it doesn't exist
    category_dir = os.path.join(dest_base_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
        continue
    
    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            # Parse the filename (X_Y.jpg)
            parts = filename.split('_')
            if len(parts) != 2:
                print(f"Warning: {filename} does not match the expected format X_Y.jpg. Skipping.")
                continue
            
            X = parts[0]
            Y_with_ext = parts[1]  # This includes .jpg
            
            # Create the X directory if it doesn't exist
            X_dir = os.path.join(category_dir, X)
            if not os.path.exists(X_dir):
                os.makedirs(X_dir)
            
            # Source and destination file paths
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(X_dir, Y_with_ext)
            
            # Move the file
            try:
                shutil.move(source_file, dest_file)
                print(f"Moved: {source_file} → {dest_file}")
            except Exception as e:
                print(f"Error moving {source_file} to {dest_file}: {e}")

print("File organization complete!") 