"""Gradio demo. Put your trained weights next to this file as best.pt, then:

    python app.py

For a free public demo, create a Hugging Face Space (Gradio) and upload app.py,
requirements.txt and best.pt.
"""
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")


def detect(image, conf):
    # Gradio gives an RGB array; Ultralytics treats numpy arrays as BGR, so flip both ways.
    result = model.predict(image[:, :, ::-1], conf=conf, verbose=False)[0]
    return result.plot()[:, :, ::-1]


demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="numpy", label="Upload a wildlife photo"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence threshold")],
    outputs=gr.Image(label="Detections"),
    title="African Wildlife Detector (YOLO)",
    description="Detects buffalo, elephant, rhino and zebra. Fine-tuned on a small public dataset, "
                "so expect mistakes on unusual angles, distant animals and poor light.",
)

if __name__ == "__main__":
    demo.launch()
