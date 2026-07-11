# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import select_device


class WorldValidator(DetectionValidator):
    """A validator for YOLO-World models that sets dataset class names before validation.

    Open-vocabulary YOLO-World models default to 80 COCO classes, so validating on a dataset with different classes
    (e.g. LVIS) fails or yields zero metrics. This validator generates text embeddings for the dataset's class names so
    standalone and training-time validation both use the active dataset vocabulary.
    """

    def __call__(self, trainer=None, model=None):
        """Set the current dataset classes for training or standalone validation."""
        if trainer is None:
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model.eval().to(self.device)
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)  # match BaseValidator dataset handling
            names = [name.split("/", 1)[0] for name in check_det_dataset(self.args.data)["names"].values()]
        else:
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in self.dataloader.dataset.data["names"].values()]

        state = (model.names, model.txt_feats, model.model[-1].nc)
        had_criterion, criterion = hasattr(model, "criterion"), getattr(model, "criterion", None)
        model.set_classes(names, cache_clip_model=False)
        model.names = dict(enumerate(names))  # set_classes updates embeddings/nc but not names
        if had_criterion:
            del model.criterion  # validation datasets can have different class counts
        try:
            return super().__call__(trainer, model)
        finally:
            model.names, model.txt_feats, model.model[-1].nc = state
            if had_criterion:
                model.criterion = criterion
            elif hasattr(model, "criterion"):
                del model.criterion
