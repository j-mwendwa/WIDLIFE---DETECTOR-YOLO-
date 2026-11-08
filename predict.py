"""Run a trained model on one image and show/save the result.

    python predict.py --weights runs/detect/wildlife/weights/best.pt --image zebra.jpg
"""
import argparse

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def prediction(model, img_path, conf=0.5, show=True, out_path=None):
    result = model.predict(img_path, conf=conf, verbose=False)[0]
    annotated = result.plot()                      # BGR image with boxes drawn
    if out_path:
        cv2.imwrite(out_path, annotated)
    if show:
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--out", default="prediction.jpg")
    a = p.parse_args()

    res = prediction(YOLO(a.weights), a.image, conf=a.conf, out_path=a.out)
    for box in res.boxes:
        print(res.names[int(box.cls)], f"{float(box.conf):.2f}")
