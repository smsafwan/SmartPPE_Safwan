from ultralytics import RTDETR
import os

def run_evaluation():
    print("===========================================")
    print("    Running Quantitative Evaluation       ")
    print("===========================================\n")

    # Load trained PyTorch weights
    model_path = 'weights/best.pt'
    model = RTDETR(model_path)

    # Run validation on validation/test split
    metrics = model.val(
        data='datasets/Custom_Eval/data.yaml',
        split='test',          # Evaluates on the 'val' split defined in data.yaml
        imgsz=800,       # Standard input resolution
        batch=2,              # Single batch size for standard evaluation
        conf=0.55,            # Standard benchmark confidence threshold (PASCAL VOC / COCO standard)
        iou=0.60,             # NMS IoU threshold
        device=0,
        workers=0,
        augment=True,
        project='custom_eval_results',
        name='metrics_new_baseline'
    )

    print("\n================ SYSTEM METRICS SUMMARY ================")
    print(f"Mean Average Precision @ IoU 0.50 (mAP50):    {metrics.box.map50:.4f}")
    print(f"Mean Average Precision @ IoU 0.50-0.95:       {metrics.box.map:.4f}")
    print(f"Overall Model Precision (Mean):               {metrics.box.mp:.4f}")
    print(f"Overall Model Recall (Mean):                  {metrics.box.mr:.4f}")
    print("========================================================\n")

    # Per-class breakdowns
    print("Per-Class mAP50 Breakdown:")
    for i, c in enumerate(metrics.box.ap50):
        class_name = metrics.names[i]
        print(f"  - {class_name:<20}: {c:.4f}")

if __name__ == '__main__':
    run_evaluation()