# African Wildlife Detector (YOLO, end to end)

Detects **buffalo, elephant, rhino and zebra** in photos. Fine-tunes a small pretrained YOLO model on the
Ultralytics African Wildlife dataset (1,504 images: 1,052 train / 225 val / 227 test), then evaluates,
exports and serves it as a demo.

> Status: code written and syntax-checked, **not yet trained**. Run it once and fill in the Results section below.

## Fastest way: Google Colab (free GPU)
1. Open `notebooks/wildlife_detector_colab.ipynb` in Colab (upload it, or open from GitHub).
2. Runtime → Change runtime type → T4 GPU.
3. Run all cells. The dataset downloads automatically.

## Or run locally / on a rented GPU
```bash
pip install -r requirements.txt
python train.py                                    # trains, then evaluates on the test split
python predict.py --weights runs/detect/wildlife/weights/best.pt --image my_photo.jpg
cp runs/detect/wildlife/weights/best.pt .          # for the demo
python app.py                                      # Gradio demo
```
Training on CPU or a Mac works but is slow; use Colab for the training run.
(The run folder name can differ by version; `train.py` prints the exact path.)

## What the notebook does
1. **Baseline:** shows the pretrained COCO model on a sample image (it lacks rhino and buffalo classes).
2. **Fine-tune:** `model.train(data="african-wildlife.yaml", epochs=50)`.
3. **Evaluate:** mAP on the held-out **test** split, per-class scores, training curves, confusion matrix.
4. **Failure analysis:** grid of test predictions to spot where it breaks.
5. **Export:** ONNX, plus `best.pt` saved to Google Drive.
6. **Demo:** Gradio app (`app.py` for a permanent Hugging Face Space).

## Results (fill in after your run)
| Metric | Value |
|---|---|
| Test mAP50 | |
| Test mAP50-95 | |
| Weakest class | |
| Inference speed (T4 / CPU) | |

## Failure analysis (fill in)
Look at the confusion matrix and 20-30 wrong predictions. Note patterns such as distant animals, herds,
partial views, low light, buffalo vs rhino confusion, and what you tried to fix them.

## Limits and honest notes
- Small dataset (about 1,000 training images, 4 classes). It will fail on unusual angles, distance and poor light.
- The dataset documentation describes animals from South African reserves; Kenyan scenes may look different.
  Adding your own Kenya photos and retraining is a strong next step.
- The Ultralytics docs list the original dataset as having no license. That is fine for learning and a
  portfolio, but check terms before redistributing the data or using the model commercially.
- Ultralytics code and models use AGPL-3.0. Check their licensing before using this in a client product.

## Next steps
- Compare `yolo11n` vs `yolo11s` and report accuracy vs speed.
- Add 50-100 of your own labeled photos and measure the change.
- Export to TFLite and test on a phone or Raspberry Pi.
- Deploy `app.py` as a Hugging Face Space and link it here.

## Share it
- Code on GitHub. Do not commit `runs/`, `datasets/` or large files.
- `best.pt` and a short model card on the Hugging Face Hub.
