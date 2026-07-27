"""Serve a YOLOE checkpoint with user-defined text prompts through Gradio.

Example:
    python examples/YOLOE-Gradio/yoloe_gradio.py \
        --model runs/detect/train/weights/best.pt \
        --text-model chineseclip:/path/to/chinese-clip-vit-base-patch16
"""

from __future__ import annotations

import argparse
import re
import threading
from collections import Counter
from typing import Any

from PIL import Image

from ultralytics import YOLOE
from ultralytics.utils.torch_utils import select_device


def parse_classes(text: str, max_classes: int = 80) -> list[str]:
    """Parse newline or comma-separated class names while preserving their order."""
    classes = list(dict.fromkeys(x.strip() for x in re.split(r"[,，;；\n]+", text or "") if x.strip()))
    if not classes:
        raise ValueError("请至少输入一个检测类别。")
    if len(classes) > max_classes:
        raise ValueError(f"最多支持 {max_classes} 个类别，当前输入了 {len(classes)} 个。")
    return classes


class YOLOEGradioService:
    """Run prompt-conditioned YOLOE inference safely for concurrent web clients."""

    def __init__(
        self,
        model_path: str,
        text_model: str | None,
        device: str,
        imgsz: int,
        max_classes: int,
        cache_text_model: bool,
    ):
        """Load the detector and configure its frozen text encoder."""
        self.device = select_device(device)
        self.model = YOLOE(model_path).to(self.device)
        if text_model:
            self.model.set_text_model(text_model)
        self.imgsz = imgsz
        self.max_classes = max_classes
        self.cache_text_model = cache_text_model
        self.lock = threading.Lock()

    def predict(self, image: Image.Image | None, prompts: str, conf: float, iou: float) -> tuple[Image.Image, str]:
        """Set the requested classes, run inference, and return the plotted image and detection counts."""
        if image is None:
            raise ValueError("请先上传一张图片。")
        classes = parse_classes(prompts, self.max_classes)

        with self.lock:
            current_classes = list(self.model.model.names.values())
            if current_classes != classes:
                embeddings = self.model.model.get_text_pe(classes, cache_clip_model=self.cache_text_model)
                self.model.set_classes(classes, embeddings)
            result = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]

        plotted = Image.fromarray(result.plot()[..., ::-1])
        counts = Counter(result.names[int(cls)] for cls in result.boxes.cls.tolist())
        summary = "未检测到目标。" if not counts else "检测结果：" + "，".join(f"{name} {count} 个" for name, count in counts.items())
        return plotted, summary


def build_interface(service: YOLOEGradioService) -> Any:
    """Build the Gradio web interface."""
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError("Gradio is required. Install it with 'pip install gradio'.") from e

    with gr.Blocks(title="YOLOE 自定义类别检测") as demo:
        gr.Markdown(
            "# YOLOE 自定义类别检测\n"
            "上传图片并输入需要检测的类别。多个类别可使用换行、中文逗号或英文逗号分隔。"
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="输入图片")
                prompts = gr.Textbox(
                    label="检测类别",
                    placeholder="例如：\n人\n小汽车\n挖掘机",
                    lines=6,
                )
                with gr.Row():
                    conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="置信度阈值")
                    iou = gr.Slider(0.01, 1.0, value=0.70, step=0.01, label="IoU 阈值")
                submit = gr.Button("开始检测", variant="primary")
            with gr.Column():
                output = gr.Image(type="pil", label="检测结果")
                summary = gr.Markdown("等待检测。")

        submit.click(
            fn=service.predict,
            inputs=[image, prompts, conf, iou],
            outputs=[output, summary],
            concurrency_limit=1,
        )
    return demo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch a prompt-conditioned YOLOE Gradio service.")
    parser.add_argument("--model", required=True, help="Path to the trained YOLOE .pt checkpoint.")
    parser.add_argument(
        "--text-model",
        default=None,
        help="Text encoder, e.g. chineseclip:/path/to/chinese-clip-vit-base-patch16. Uses checkpoint config if omitted.",
    )
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0, 1, or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-classes", type=int, default=80, help="Maximum class prompts accepted per request.")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address.")
    parser.add_argument("--port", type=int, default=9009, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL.")
    parser.add_argument(
        "--no-cache-text-model",
        dest="cache_text_model",
        action="store_false",
        help="Reload the text encoder for each changed prompt to reduce persistent memory usage.",
    )
    parser.set_defaults(cache_text_model=True)
    return parser.parse_args()


def main() -> None:
    """Load models and launch the Gradio service."""
    args = parse_args()
    service = YOLOEGradioService(
        model_path=args.model,
        text_model=args.text_model,
        device=args.device,
        imgsz=args.imgsz,
        max_classes=args.max_classes,
        cache_text_model=args.cache_text_model,
    )
    build_interface(service).queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
