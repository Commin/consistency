from pathlib import Path
from ultralytics import YOLO
import json
import argparse

def run_test_val(model_path, target_data_yaml, project, val_name):
    print(f"\n>>> [TEST VAL] Model: {model_path} on {target_data_yaml}")
    model = YOLO(model_path)
    metrics = model.val(
        data=target_data_yaml,
        split='test', 
        save_txt=True,
        save_conf=True,
        project=project,
        name=val_name,
        device='0'
    )
    return metrics

def save_metrics(metrics, metrics_path):

    offline_data = {
        "summary": metrics.results_dict,
        "speed_ms": metrics.speed,
        "per_class_map": metrics.box.maps.tolist()
    }
    with open(metrics_path, 'w') as f:
        json.dump(offline_data, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="run", help="Root directory for saving results and constructed datasets.")
    parser.add_argument("--mode", type=str, default="sample", choices=["all_data", "sample", "confidence", "consistency"], help="Learning mode for dataset construction.")
    parser.add_argument("--stages", type=list, help="The index of the stage to start from (default: 0). This is useful for resuming from a specific stage in case of interruptions.")
    parser.add_argument("--data", type=str, help="Path to the data YAML file.")
    args = parser.parse_args()

    results_root = f"{args.project}/{args.mode}"
    target_data_yaml = args.data
    
    # Modify the stages list according to your actual directory structure under results_root.
    # stages = [
    #     "clear_00",
    #     "clear_10",
    #     "clear_20",
    #     "clear_30",
    #     "clear_40",
    #     "clear_50"
    # ]
    
    project = results_root
    
    for name in args.stages:
        model_path = str(Path(project) / name / "weights" / "best.pt")
        
        # Set the validation name to be unique for each stage, e.g., "test_val_00", "test_val_10", etc. This will help in organizing the results and avoiding overwriting.
        val_name = f"test_val_{name}"
        
        if not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
            
        try:
            metrics = run_test_val(
                model_path=model_path,
                target_data_yaml=target_data_yaml,
                project=project,
                val_name=val_name
            )

            # save metrics to json file
            metrics_path = Path(project) / f"test_metrics_{name}.json"
            save_metrics(metrics, metrics_path)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error during validation for {name}: {e}")

if __name__ == "__main__":
    main()
