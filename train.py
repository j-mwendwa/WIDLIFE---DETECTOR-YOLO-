"""Fine-tune a small YOLO model on the Ultralytics African Wildlife dataset.

The dataset (1,504 images: buffalo, elephant, rhino, zebra) downloads automatically
the first time you train. A GPU is strongly recommended (free Colab T4 is enough).

    python train.py                      # defaults: yolo11n, 50 epochs
    python train.py --model yolo11s.pt --epochs 80
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")   # try yolo26n.pt if your ultralytics version has it
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(data="african-wildlife.yaml", epochs=a.epochs, imgsz=a.imgsz,
                batch=a.batch, name="wildlife")

    save_dir = Path(model.trainer.save_dir)
    best = model.trainer.best
    print(f"\nRun folder: {save_dir}\nBest weights: {best}")

    # Honest final numbers: evaluate on the held-out TEST split.
    metrics = YOLO(best).val(data="african-wildlife.yaml", split="test")
    print(f"Test mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    print("Per-class mAP50-95:", dict(zip(metrics.names.values(), metrics.box.maps.round(3))))


if __name__ == "__main__":
    main()
