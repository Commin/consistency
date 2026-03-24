import yaml
import random
import pandas as pd
from pathlib import Path

def generate_stage_yaml(stages, current_idx, project_path):
    """
    According to the current stage index, merge the data from all previous stages and generate a new data.yaml.
    """
    
    current_stage = stages[current_idx]
    stage_name = current_stage['name']
    
    # Save to project_path / name
    save_dir = Path(project_path) / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_path = save_dir / "data.yaml"

    combined_train = []
    combined_val = []
    
    # Configuration for the current stage's metadata (nc, names) will be taken from the current stage's yaml, but paths will be merged from all stages up to current_idx.
    current_yaml_file = Path(stages[current_idx]['data'])
    with open(current_yaml_file, 'r', encoding='utf-8') as f:
        current_data = yaml.safe_load(f)
        
    master_path_str = current_data.get('path')
    if not master_path_str:
        master_path_str = str(current_yaml_file.parent)
    master_path = Path(master_path_str).resolve()
    
    final_meta_nc = current_data.get('nc')
    final_meta_names = current_data.get('names')

    # Traverse all yaml files from 0 to current_idx and merge their train/val paths
    for j in range(current_idx + 1):
        yaml_file = Path(stages[j]['data'])
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        base_path_str = data.get('path')
        if not base_path_str:
            base_path_str = str(yaml_file.parent)
        base_path = Path(base_path_str).resolve()
        
        # Path conversion: Convert all paths to absolute, then to relative to master_path if possible
        for key in ['train', 'val']:
            paths = data.get(key, [])
            if isinstance(paths, str):
                paths = [paths]
            
            for p in paths:
                p_path = Path(p)
                
                if not p_path.is_absolute():
                    full_p = (base_path / p_path).resolve()
                else:
                    full_p = p_path.resolve()
                
                try:
                    rel_p = str(full_p.relative_to(master_path))
                except ValueError:
                    rel_p = str(full_p)
                
                if key == 'train':
                    if rel_p not in combined_train:
                        combined_train.append(rel_p)
                else:
                    if rel_p not in combined_val:
                        combined_val.append(rel_p)

    # Save to new yaml
    new_data = {
        'path': str(master_path),
        'train': combined_train if len(combined_train) != 1 else (combined_train[0] if combined_train else []),
        'val': combined_val if len(combined_val) != 1 else (combined_val[0] if combined_val else []),
        'nc': final_meta_nc,
        'names': final_meta_names
    }

    # Write the new yaml file
    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_data, f, sort_keys=False, allow_unicode=True)
    
    print(f"✅ Generated Stage {current_idx} ({stage_name}) data yaml file: {new_yaml_path}")
    return new_yaml_path

# --- Example Usage ---
# project = "/home/user/my_yolo_project"
# for i in range(len(stages)):
#     generate_stage_yaml(stages, i, project)


def construct_dataset(mode, base_yaml, val_results_dir, stage_name, project_root):
    """
    Third stage: Construct dataset. Generate dedicated txt and yaml according to the mode.
    """

    all_imgs = get_image_paths_from_yaml(base_yaml, split='train')
    print(f"\n>>> [CONSTRUCT] Mode: {mode} for {stage_name}, all_imgs: {len(all_imgs)}")
    selected_imgs = []
    
    if mode == "all_data":
        return base_yaml

    elif mode == "sample":
        # Randomly sample 10%
        sample_size = max(1, int(len(all_imgs) * 0.1))
        selected_imgs = random.sample(all_imgs, sample_size)

    elif mode == "confidence":
        # 1. Measure object-level confidence and save as json
        frame_scores_csv, object_details_json = evaluate_confidence_and_save(
            all_imgs, 
            val_results_dir, 
            stage_name, 
            output_dir=Path(project_root) / "custom_datasets"
        )
        
        # 2. Measure frame-level confidence and save as csv, which will be used for balanced sampling in the next step.
        # Warning: The logic for selecting frames based on confidence is the same as consistency, where lower confidence indicates more unstable/harder samples. We will use balance_and_select_frames to select the worst 10%.

        selected_imgs = balance_and_select_frames(
            frame_scores_csv, 
            total_images=len(all_imgs), 
            ratio=0.1, 
            max_consecutive=2,
            score_by="score"
        )

    elif mode == "consistency":
        # 1. Measure object-level consistency and save as json
        # Calculate frame-level consistency and save as csv at the same time, which will be used for balanced sampling in the next step.

        frame_scores_csv, object_details_json = evaluate_consistency_and_save(
            all_imgs, 
            val_results_dir, 
            stage_name, 
            output_dir=Path(project_root) / "custom_datasets",
            delete_path_suffix=False
        )
        
        # 2. Read frame-level consistency results, balance the data and select the worst 10%
        selected_imgs = balance_and_select_frames(
            frame_scores_csv, 
            total_images=len(all_imgs), 
            ratio=0.1, 
            max_consecutive=2,
            score_by="frame_consistency"
        )

        # 3. if selected_imgs only contain stem without path or .txt, need to complete the path
        with open(base_yaml, 'r') as f:
            base_data = yaml.safe_load(f)
        base_path = Path(base_data['path'])
        if all(isinstance(x, str) and not x.endswith(('.txt', '.jpg', '.png', '.jpeg')) for x in selected_imgs):
            selected_imgs = [str(base_path / "train" / "images" / f"{stem}.jpg") for stem in selected_imgs]

    # --- Generate Output Files ---
    output_dir = Path(project_root) / "custom_datasets"
    output_dir.mkdir(exist_ok=True)
    
    txt_path = output_dir / f"{stage_name}_{mode}.txt"
    with open(txt_path, 'w') as f:
        f.write('\n'.join(selected_imgs))
        
    # New YAML
    with open(base_yaml, 'r') as f:
        new_config = yaml.safe_load(f)
    
    new_config['train'] = str(txt_path.absolute())
    new_yaml_path = output_dir / f"{stage_name}_{mode}.yaml"
    with open(new_yaml_path, 'w') as f:
        yaml.safe_dump(new_config, f)
        
    return str(new_yaml_path)

def evaluate_confidence_and_save(all_imgs, val_results_dir, stage_name, output_dir):
    """
    Calculate and save confidence:
    - frame-level CSV: key/path = current frame T, value = minimum confidence of all objects in this frame, representing the confidence of transition T -> T+1
    - object-level JSON: key/path = current frame T, value = list of all objects with their confidence and details 
    """
    import json
    import consistency
    from pathlib import Path
    import pandas as pd
    
    labels_dir = Path(val_results_dir) / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{stage_name}_frame_confidence.csv"
    json_path = output_dir / f"{stage_name}_object_confidence.json"
    
    frame_data = []
    object_details_dict = {}
    
    # Sort images by video sequence and frame number to ensure the rationality of consecutive sampling
    def extract_sequence_and_frame(path_str):
        name = Path(path_str).stem
        parts = name.split('_frame')
        if len(parts) == 2:
            seq = parts[0]
            try:
                frame_num = int(parts[1].split('_')[0].split('.')[0])
                return seq, frame_num
            except ValueError:
                return seq, -1
        return name, -1
        
    sorted_imgs = sorted(all_imgs, key=extract_sequence_and_frame)

    for i, img_path in enumerate(sorted_imgs):
        txt_f = labels_dir / f"{Path(img_path).stem}.txt"
        preds = consistency.parse_yolo_txt(str(txt_f))
        
        objects_conf = []
        frame_min_conf = 1.0 # Default confidence if there are objects, will be updated to the minimum confidence among objects. If no objects, will be set to 0.0 later.
        
        if len(preds) > 0:
            for p in preds:
                # if parse_yolo_txt exists confidence, it will be at index 5
                # [class_id, cx, cy, w, h, confidence]
                conf = float(p[5]) if len(p) > 5 else 1.0 # If validation set does not output confidence, default to 1.0
                objects_conf.append({
                    "class_id": int(p[0]),
                    "confidence": conf,
                    "box": p[1:5]
                })
            
            # frame-level confidence is the minimum confidence among all objects in this frame, representing the confidence of transition T -> T+1
            frame_min_conf = min([obj["confidence"] for obj in objects_conf])
        else:
            # if there are no predicted bounding boxes in the whole frame (possible missed detection), consider its confidence as 0.0, giving it the highest priority for re-learning
            frame_min_conf = 0.0
            
        frame_data.append({
            "path": img_path,
            "score": frame_min_conf,
            "original_index": i
        })
        
        object_details_dict[img_path] = {
            "frame_score": frame_min_conf,
            "objects": objects_conf
        }
        
    # Save CSV
    df = pd.DataFrame(frame_data)
    df.to_csv(csv_path, index=False)
    
    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(object_details_dict, f, indent=4)
        
    print(f"Saved frame-level confidence to: {csv_path}")
    print(f"Saved object-level confidence to: {json_path}")
    
    return csv_path, json_path

def evaluate_consistency_and_save(all_imgs, val_results_dir, stage_name, output_dir, delete_path_suffix=True):
    """
    Calculate consistency based on the validation results and save the results in both frame-level CSV and object-level JSON formats. The consistency evaluation is based on the transition between consecutive frames, so the key/path in both CSV and JSON represents the current frame T, which corresponds to the transition from T to T+1.
    - The frame-level CSV contains the consistency score for each transition (T -> T+1), where the key/path is the current frame T.
    - The object-level JSON contains detailed information about the matched objects between frame T and T+1, including their confidence scores and matching details. The key/path is also the current frame T. This allows us to analyze not only the overall consistency of the frame transition but also the specific objects that contribute to the consistency score, which can be useful for targeted retraining in the next stage.
    """

    import json
    import consistency
    from pathlib import Path
    import pandas as pd

    labels_dir = Path(val_results_dir) / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stage_name}_frame_consistency.csv"
    json_path = output_dir / f"{stage_name}_object_consistency.json"

    frame_data = []
    object_details_dict = {}

    # ---------- Path Processing ----------
    def clean_path(path_str):
        if not delete_path_suffix:
            return path_str

        name = Path(path_str).name   # remove directory
        name = name.rsplit(".", 1)[0]  # remove last suffix (.jpg / .png)
        return name
    # -----------------------------------

    # Sort video frames by video sequence and frame number to ensure the rationality of consecutive sampling
    def extract_sequence_and_frame(path_str):
        name = Path(path_str).stem
        parts = name.split("_frame")
        if len(parts) == 2:
            seq = parts[0]
            try:
                frame_num = int(parts[1].split("_")[0].split(".")[0])
                return seq, frame_num
            except ValueError:
                return seq, -1
        return name, -1

    sorted_imgs = sorted(all_imgs, key=extract_sequence_and_frame)

    for i in range(len(sorted_imgs) - 1):

        curr_img_path = sorted_imgs[i]
        next_img_path = sorted_imgs[i + 1]

        seq_curr, _ = extract_sequence_and_frame(curr_img_path)
        seq_next, _ = extract_sequence_and_frame(next_img_path)

        consistency_score = 1.0
        mean_match_quality = 1.0
        match_pairs = []

        if seq_curr == seq_next:

            txt_f_curr = labels_dir / f"{Path(curr_img_path).stem}.txt"
            txt_f_next = labels_dir / f"{Path(next_img_path).stem}.txt"

            consistency_score, mean_match_quality, match_pairs = \
                consistency.evaluate_consistency_from_files(
                    str(txt_f_curr),
                    str(txt_f_next)
                )

        curr_key = clean_path(curr_img_path)
        next_key = clean_path(next_img_path)

        frame_data.append({
            "path": curr_key,
            "next_path": next_key,
            "frame_consistency": consistency_score,
            "original_index": i,
        })

        object_details_dict[curr_key] = {
            "score": mean_match_quality,
            "matches": match_pairs,
            "transition_from": curr_key,
            "transition_to": next_key,
        }

    # Add the last frame, which does not have a next frame for transition, we can set its consistency score to 1.0 by default, and the object-level details can be empty or default values. This ensures that every frame in the dataset has a corresponding entry in both the CSV and JSON files, which can simplify downstream processing and analysis.
    if sorted_imgs:

        last_img_path = sorted_imgs[-1]
        last_key = clean_path(last_img_path)

        frame_data.append({
            "path": last_key,
            "next_path": None,
            "frame_consistency": 1.0,
            "original_index": len(sorted_imgs) - 1,
        })

        object_details_dict[last_key] = {
            "score": 1.0,
            "matches": [],
            "transition_from": last_key,
            "transition_to": None,
        }

    df = pd.DataFrame(frame_data)
    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(object_details_dict, f, indent=4, ensure_ascii=False)

    print(f"Saved frame-level consistency to: {csv_path}")
    print(f"Saved object-level consistency to: {json_path}")

    return csv_path, json_path

def balance_and_select_frames(csv_path, total_images, ratio=0.1, max_consecutive=2, score_by="frame_consistency"):
    """
    Select frames based on the frame_consistency results, choosing the worst part, while ensuring that no more than max_consecutive frames are selected in a row.
    """
    import pandas as pd
    
    target_count = max(1, int(total_images * ratio))
    df = pd.read_csv(csv_path)
    
    # Sort by the specified score (e.g., frame_consistency), from low to high (lower score means more unstable, which is what we want to sample)
    df_sorted = df.copy().sort_values(by=score_by, ascending=True)
    
    selected_paths = []
    selected_indices = set() # Save records of already selected original_index
    
    for _, row in df_sorted.iterrows():
        if len(selected_paths) >= target_count:
            break
            
        current_idx = int(row['original_index'])
        path = row['path']
        
        # Check if this frame has already been selected (in case of duplicates)
        consecutive_before = 0
        idx_check = current_idx - 1
        while idx_check in selected_indices:
            consecutive_before += 1
            idx_check -= 1
            
        consecutive_after = 0
        idx_check = current_idx + 1
        while idx_check in selected_indices:
            consecutive_after += 1
            idx_check += 1
            
        # If adding the current frame would exceed the max_consecutive limit, skip it
        if consecutive_before + consecutive_after + 1 > max_consecutive:
            continue
            
        # If the current frame meets the criteria, add it to the selected list
        selected_paths.append(path)
        selected_indices.add(current_idx)
        
    # If due to the consecutive constraint the final selection is less than target_count,
    # decide whether to relax the constraint to fill the remaining slots, here we choose to forcefully relax the condition to fill up to target_count
    if len(selected_paths) < target_count:
        print(f"Warning: Could only strictly select {len(selected_paths)}/{target_count} due to consecutive constraint.")
        print("Relaxing constraint to fill the remaining slots...")
        for _, row in df_sorted.iterrows():
            if len(selected_paths) >= target_count:
                break
            if row['path'] not in selected_paths:
                selected_paths.append(row['path'])
                selected_indices.add(int(row['original_index']))
                
    print(f"Selected {len(selected_paths)} frames for construct mode.")
    return selected_paths

def get_image_paths_from_yaml(target_data_yaml, split='train'):
    with open(target_data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    paths = cfg.get(split)
    if not paths:
        return []
        
    base_path_str = cfg.get('path', str(Path(target_data_yaml).parent))
    base_path = Path(base_path_str).resolve()
        
    if isinstance(paths, str):
        paths = [paths]
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_image_paths = []
    
    for p in paths:
        p_path = Path(p)
        if not p_path.is_absolute():
            p_path = (base_path / p_path).resolve()
            
        if p_path.suffix.lower() == '.txt':
            if p_path.exists():
                with open(p_path, 'r', encoding='utf-8') as txt_file:
                    for line in txt_file:
                        line = line.strip()
                        if line:
                            all_image_paths.append(line)
        else:
            if p_path.name == 'images':
                search_dir = p_path
            else:
                search_dir = p_path / 'images'
                
            if search_dir.exists() and search_dir.is_dir():
                for fname in search_dir.iterdir():
                    if fname.is_file() and fname.suffix.lower() in valid_extensions:
                        all_image_paths.append(str(fname.absolute()))
                        
    return all_image_paths