import cv2
import os
import glob
import re
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def parse_sort_key(path: str):
    """
    Compare the file names based on the following rules:

    For example:
      MVI_0790_VIS_OB_frame0_jpg.rf.xxx.jpg   -> ('MVI_0790_VIS_OB', 0)
      MVI_0790_VIS_OB_frame10_jpg.rf.xxx.jpg  -> ('MVI_0790_VIS_OB', 10)
      MVI_0791_VIS_OB_frame5_jpg.rf.xxx.jpg   -> ('MVI_0791_VIS_OB', 5)
    """
    name = os.path.basename(path)

    # Compare the prefix before '_frame' first, then compare the number between '_frame' and '_jpg' if the prefix is the same.
    m = re.match(r"^(.*?)_frame(\d+)_jpg", name)
    if m:
        prefix = m.group(1)
        frame_idx = int(m.group(2))
        return (prefix, frame_idx, name)

    # If the filename does not match the pattern, put it at the end and keep the original filename as a fallback sorting key.
    return (name, float("inf"), name)



def extract_video_prefix(path: str):
    name = os.path.basename(path)
    m = re.match(r"^(.*?)_frame\d+_jpg", name)
    if m:
        return m.group(1)
    return None



def extract_frame_index(path: str):
    name = os.path.basename(path)
    m = re.match(r"^.*?_frame(\d+)_jpg", name)
    if m:
        return int(m.group(1))
    return None



def compute_folder_ssim(img_dir, output_csv):
    """
    Traverse the folder of images and compute SSIM between adjacent frames of the same video/sequence.
    1) Sort the images first by the prefix before '_frame', then by the number between '_frame' and '_jpg' when the prefix is the same.
    2) Only compute SSIM between adjacent frames with the same prefix to avoid incorrect comparisons between different videos/sequences.

    Only compute SSIM between adjacent frames with the same prefix to avoid incorrect comparisons between different videos/sequences.
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))

    image_files = sorted(image_files, key=parse_sort_key)

    if len(image_files) < 2:
        print(f"Error: Folder {img_dir} does not contain enough images to compute SSIM.")
        return

    print(f"Found {len(image_files)} images, starting computation...")
    results = []

    for i in range(len(image_files) - 1):
        p1 = image_files[i]
        p2 = image_files[i + 1]

        prefix1 = extract_video_prefix(p1)
        prefix2 = extract_video_prefix(p2)

        # Only compare adjacent frames within the same sequence to avoid incorrect comparisons between different videos/sequences.
        if prefix1 != prefix2:
            continue

        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)

        if img1 is None or img2 is None:
            print(f"Skipping: Failed to read {p1} or {p2}")
            continue

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        score = ssim(img1, img2, channel_axis=-1)

        results.append({
            "file_name": os.path.basename(p1),
            "next_file_name": os.path.basename(p2),
            "prefix": prefix1,
            "frame_idx": extract_frame_index(p1),
            "next_frame_idx": extract_frame_index(p2),
            "ssim_value": score,
        })

        if len(results) % 100 == 0:
            print(f"Progress: {len(results)} pairs of images processed...")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n--- Calculation Complete ---")
    print(f"Total Results: {len(results)}")
    print(f"Output File: {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv-name", type=str, default="output_ssim.csv", help="Name of the output CSV file (default: output_ssim.csv)")

    args = parser.parse_args()

    IMAGE_DIRECTORY = args.gt_dir
    OUTPUT_FILE = os.path.join(args.output_dir, args.csv_name)  

    compute_folder_ssim(IMAGE_DIRECTORY, OUTPUT_FILE)
