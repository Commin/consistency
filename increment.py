import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import torch

from construct import construct_dataset, get_image_paths_from_yaml, generate_stage_yaml


def run_train_stage(model_path, data_yaml, project, name, batch, epochs, train, freeze=None):
    if train:
        print(f"\n>>> [TRAIN] Stage: {name} | Model: {model_path}")
        model = YOLO(model_path)
        
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch,
            'imgsz': 640,
            'project': project,
            'name': name,
            'device': '0',
            'val': True,
            'exist_ok': True,
            'cache': False 
        }
        if freeze is not None:
            train_args['freeze'] = freeze
            
        results = model.train(**train_args)
    else:
        print(f"\n>>> [TRAIN] Stage: No training.....")

    torch.cuda.empty_cache()
    return str(Path(project) / name / "weights" / "best.pt")

def run_val_stage(model_path, target_data_yaml, project, val_name):
    print(f"\n>>> [VAL] Model: {model_path} on {target_data_yaml}")
    model = YOLO(model_path)
    # Validate on the target dataset's training split to get predictions for constructing the next stage's training set
    metrics = model.val(
        data=target_data_yaml,
        split='train', 
        save_txt=True,
        save_conf=True,
        project=project,
        name=val_name,
        device='0'
    )
    
    # Add empty TXT files for any images that did not have predictions, to ensure that the constructed dataset for the next stage has a complete set of TXT files corresponding to all images in the target dataset, which is important for consistent data handling and training in the subsequent stage.
    save_dir = Path(metrics.save_dir)
    labels_dir = save_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    img_list = get_image_paths_from_yaml(target_data_yaml, split='train')
    
    for img_path in img_list:
        txt_path = labels_dir / f"{Path(img_path).stem}.txt"
        if not txt_path.exists():
            txt_path.touch()
            
    return str(save_dir) # Return the directory where the validation results (TXT files) are saved, which will be used for constructing the next stage's dataset

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally train YOLOv12 models."
    )
    parser.add_argument("--project", type=str, default="run", help="Root directory for saving results and constructed datasets.")
    parser.add_argument("--mode", type=str, default="sample", choices=["all_data", "sample", "confidence", "consistency"], help="Learning mode for dataset construction.")
    parser.add_argument("--train", action="store_true", help="Whether to perform training at each stage.")
    parser.add_argument("--increment", action="store_true", help="Whether to use the incrementally constructed dataset for the next stage, or always use the original split.")
    parser.add_argument("--no_init", action="store_true", help="Whether to skip training for the initial stage (stage 0) and directly use the existing weights, which is useful if the initial stage has already been trained and we want to save time.")
    parser.add_argument("--stage", type=int, default=0, help="The index of the stage to start from (default: 0). This is useful for resuming from a specific stage in case of interruptions.")

def main():
    args = build_argparser().parse_args()
    LEARNING_MODE = args.mode
    TRAIN = args.train
    INCREMENT = args.increment
    NO_INIT = args.no_init
    start_idx = args.stage
    
    stages = [
        {"name": "clear_00", "data": "data0.yaml", "freeze": None},
        {"name": "clear_10", "data": "data1.yaml", "freeze": 10},
        {"name": "clear_20", "data": "data2.yaml", "freeze": 10},
        {"name": "clear_30", "data": "data3.yaml", "freeze": 10},
        {"name": "clear_40", "data": "data4.yaml", "freeze": 10},
        {"name": "clear_50", "data": "data5.yaml", "freeze": 10},
    ]
    
    results_root = os.path.join(args.project, f"{LEARNING_MODE}")
    
    # Initial weight and data setup
    current_weight = "yolov12x.pt" 
    current_data_yaml = stages[0]['data']

    # if start_idx > 0, it means we are resuming from a specific stage, so we need to set the current_weight and current_data_yaml according to that stage's expected inputs.
    if start_idx > 0:
        # best.pt in the previous stage is the expected starting weight for the current stage
        current_weight = str(Path(results_root) / stages[start_idx-1]['name'] / "weights" / "best.pt")
        
        # data.yaml for the current stage is determined by whether we are using incremental datasets or always using the original splits
        if INCREMENT:
            base_yaml_for_resume = str(Path(results_root) / stages[start_idx]['name'] / "data.yaml")
        else:
            base_yaml_for_resume = stages[start_idx]['data']
            
        if LEARNING_MODE == "all_data":
            current_data_yaml = base_yaml_for_resume
        else:
            current_data_yaml = str(Path(results_root) / "custom_datasets" / f"{stages[start_idx]['name']}_{LEARNING_MODE}.yaml")
            
        print(f"\n>>> [RESUME] Starting from stage {start_idx} ({stages[start_idx]['name']})")
        print(f">>> [RESUME] Using weight: {current_weight}")
        print(f">>> [RESUME] Using data: {current_data_yaml}\n")
        
        if not Path(current_weight).exists():
            raise FileNotFoundError(f"Resume failed: Missing weights file -> {current_weight}")
        if not Path(current_data_yaml).exists():
            raise FileNotFoundError(f"Resume failed: Missing data file -> {current_data_yaml}")

    for i in range(start_idx, len(stages)):
        curr = stages[i]
        
        # 1. TRAIN: use the current data and weights to train the model for this stage
        if i == 0 and NO_INIT:
            # if it's the initial stage and we choose to skip training, directly set its output weights as the starting point for subsequent stages
            current_weight = str(Path(results_root) / curr['name'] / "weights" / "best.pt")
            print(f"\n>>> [TRAIN] Skip training for init stage: {curr['name']}, using existing weights at: {current_weight}")
        else:
            current_weight = run_train_stage(
                model_path=current_weight,
                data_yaml=current_data_yaml,
                project=results_root,
                name=curr['name'],
                batch=8,
                epochs=100,
                train=TRAIN,
                freeze=curr['freeze']
            )
        
        # 2. VAL and 3. CONSTRUCT will be done together after the training loop, because the validation results are needed for constructing the next stage's dataset, and we want to ensure that the training is completed before moving on to validation and construction.
        if i < len(stages) - 1:
            next_stage = stages[i+1]
            
            val_dir = run_val_stage(
                model_path=current_weight,
                target_data_yaml=next_stage['data'],
                project=results_root,
                val_name=f"pre_val_{next_stage['name']}"
            )
            

            # Construct the dataset for the next stage based on the validation results and the specified learning mode. The constructed dataset will be saved as a new data.yaml file, and its path will be used as the current_data_yaml for the next iteration.
            if INCREMENT:
                base_yaml_for_construct = generate_stage_yaml(
                    stages=stages, 
                    current_idx=i + 1, 
                    project_path=results_root
                )
            else:
                base_yaml_for_construct = next_stage['data']
                
            current_data_yaml = construct_dataset(
                mode=LEARNING_MODE,
                base_yaml=base_yaml_for_construct,
                val_results_dir=val_dir,
                stage_name=next_stage['name'],
                project_root=results_root
            )

if __name__ == "__main__":
    main()
