import os
import numpy as np
import cv2
import argparse
from datetime import datetime
import logging
from pathlib import Path
import ncempy.io as nio  # For reading DM4 files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_argparse():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extract and process in-situ video frames from F200 microscope DM4 files')
    parser.add_argument('--source_dir', type=str, help='Source directory containing F200 microscope data',
                        default="/Users/vnpawan/Documents/03.07.25")
    parser.add_argument('--output_dir', type=str, help='Output directory for extracted frames', default="./output")
    parser.add_argument('--create_video', action='store_true', help='Create video from extracted frames')
    parser.add_argument('--video_fps', type=int, default=10, help='Frames per second for output video')
    parser.add_argument('--contrast_adjust', action='store_true', help='Adjust contrast of extracted frames')
    return parser.parse_args()


def extract_time_info(file_path):
    """Extract hour, minute, second, and frame directly from path components"""
    path_parts = str(file_path).split(os.sep)

    # Initialize with default values
    hour, minute, second, frame = None, None, None, None

    # Extract directly from path parts
    for part in path_parts:
        if part.startswith("Hour_"):
            hour = int(part[5:])
        elif part.startswith("Minute_"):
            minute = int(part[7:])
        elif part.startswith("Second_"):
            second = int(part[7:])

    # Extract frame from filename
    filename = os.path.basename(file_path)
    if "Frame_" in filename:
        frame_str = filename.split("Frame_")[1].split(".")[0]
        frame = int(frame_str)

    return hour, minute, second, frame


def read_dm4_file(file_path):
    """Read a DM4 file and return the image data"""
    try:
        # Use ncempy to read DM4 file
        with nio.dm.fileDM(file_path) as dm_file:
            # Read the data and metadata
            dataset = dm_file.getDataset(0)
            data = dataset['data']

            # Convert to numpy array and normalize to 0-255 range for image
            image_data = data.astype(np.float32)

            # Normalize the data
            if image_data.min() != image_data.max():
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
            image_data = image_data.astype(np.uint8)

            return image_data
    except Exception as e:
        logger.error(f"Error reading DM4 file {file_path}: {str(e)}")
        return None


def adjust_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply contrast limited adaptive histogram equalization to improve image contrast"""
    if len(image.shape) == 3:  # Color image
        image_yuv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV)
        image_yuv[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(
            image_yuv[:, :, 0])
        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    else:  # Grayscale image
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(image.astype(np.uint8))


def extract_frames(source_dir, output_dir, contrast_adjust=False):
    """Extract frames from the F200 microscope directory structure with DM4 files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_data = []  # Store (timestamp, file_path) for later sorting

    # Walk through the directory structure
    for root, dirs, files in os.walk(source_dir):
        # Process DM4 files in this directory
        for file in files:
            if file.lower().endswith('.dm4'):
                file_path = os.path.join(root, file)

                # Extract time information from file path
                hour, minute, second, frame = extract_time_info(file_path)

                if hour is not None and minute is not None and second is not None:
                    # Create a timestamp for sorting - use seconds for sequential ordering
                    timestamp = (hour * 3600) + (minute * 60) + second

                    # Add frame number as a fractional part for precise ordering within each second
                    if frame is not None:
                        timestamp += frame / 10000.0

                    frame_data.append((timestamp, file_path))

                    if len(frame_data) % 100 == 0:
                        logger.info(f"Found {len(frame_data)} DM4 files so far")

    # Sort by timestamp to ensure chronological order with seconds
    frame_data.sort(key=lambda x: x[0])
    logger.info(f"Found {len(frame_data)} total DM4 files, sorted in chronological order")

    # Process and save frames
    for frame_idx, (timestamp, file_path) in enumerate(frame_data):
        try:
            # Read DM4 file
            image_data = read_dm4_file(file_path)

            if image_data is not None:
                # Apply contrast adjustment if requested
                if contrast_adjust:
                    image_data = adjust_contrast(image_data)

                # Save as image file
                output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.tiff")
                cv2.imwrite(output_path, image_data)

                if frame_idx % 10 == 0:
                    logger.info(f"Processed {frame_idx + 1}/{len(frame_data)} frames")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    logger.info(f"Extracted {len(frame_data)} frames in chronological order")
    return len(frame_data)


def create_video(frames_dir, output_video, fps=10):
    """Create a video from the extracted frames"""
    frames = [f for f in os.listdir(frames_dir) if f.endswith(('.tiff', '.jpg', '.png'))]
    frames.sort()

    if not frames:
        logger.error("No frames found to create video")
        return False

    # Get frame dimensions from first image
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first_frame is None:
        logger.error(f"Could not read first frame: {os.path.join(frames_dir, frames[0])}")
        return False

    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = output_video
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Add frames to video
    frame_count = 0
    for frame_file in frames:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame is not None:
            video_writer.write(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Added {frame_count}/{len(frames)} frames to video")

    video_writer.release()
    logger.info(f"Created video with {frame_count} frames at {fps} FPS: {video_path}")
    return True


def main():
    """Main function to run the script"""
    # Get command line arguments
    args = setup_argparse()

    source_dir =
    output_dir =

    print(f"\nProcessing F200 microscope data from: {source_dir}")
    print(f"Output will be saved to: {output_dir}")

    # Extract frames
    print("\nExtracting frames...")
    num_frames = extract_frames(source_dir, output_dir, args.contrast_adjust)

    # Create video if requested
    if args.create_video and num_frames > 0:
        video_filename = f"f200_timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        print(f"\nCreating video at: {video_path}")
        create_video(output_dir, video_path, args.video_fps)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()