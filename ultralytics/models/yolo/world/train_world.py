# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch.distributed as dist

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOCAL_RANK, LOGGER
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class WorldTrainerFromScratch(WorldTrainer):
    """A class extending the WorldTrainer for training a world model from scratch on open-set datasets.

    This trainer specializes in handling mixed datasets including both object detection and grounding datasets,
    supporting training YOLO-World models with combined vision-language capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with default parameters for model training.
        overrides (dict): Dictionary of parameter overrides to customize the configuration.
        _callbacks (dict): Dictionary of callback functions to be executed during different stages of training.
        data (dict): Final processed data configuration containing train/val paths and metadata.
        training_data (dict): Dictionary mapping training dataset paths to their configurations.

    Methods:
        build_dataset: Build YOLO Dataset for training or validation with mixed dataset support.
        get_dataset: Get train and validation paths from data dictionary.
        plot_training_labels: Skip label plotting for YOLO-World training.
        final_eval: Perform final evaluation and validation for the YOLO-World model.

    Examples:
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="flickr30k/images",
        ...                 json_file="flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="GQA/images",
        ...                 json_file="GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize a WorldTrainerFromScratch object.

        This initializes a trainer for YOLO-World models from scratch, supporting mixed datasets including both object
        detection and grounding datasets for vision-language capabilities.

        Args:
            cfg (dict): Configuration dictionary with default parameters for model training.
            overrides (dict, optional): Dictionary of parameter overrides to customize the configuration.
            _callbacks (dict, optional): Dictionary of callback functions to run during different stages of training.
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset for training or validation.

        This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO
        datasets and grounding datasets with different formats.

        Args:
            img_path (list[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(
                self.args,
                im_path,
                batch,
                self.training_data[im_path],
                stride=gs,
                multi_modal=True,
                max_samples=self.data["max_text_samples"],
            )
            if isinstance(im_path, str)
            else build_grounding(
                # Use the same text capacity for YOLO and grounding datasets.
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["max_text_samples"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)  # cache text embeddings to accelerate training
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        """Check and load the data configuration from a YAML file or dictionary.

        Args:
            data (dict | str | Path): Data configuration as a dictionary or path to a YAML file.

        Returns:
            (dict): Data configuration dictionary loaded from YAML file or passed directly.
        """
        # If string, load from YAML file
        if not isinstance(data, dict):
            from ultralytics.utils import YAML

            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        """Get train and validation paths from data dictionary.

        Processes the data configuration to extract paths for training and validation datasets, handling both YOLO
        detection datasets and grounding datasets.

        Returns:
            (dict): Final processed data configuration containing train/val paths and metadata.

        Raises:
            AssertionError: If train or validation datasets are not found.
        """
        self.args.data = data_yaml = self.check_data_config(self.args.data)
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        train_sources = data_yaml["train"].get("yolo_data", [])
        train_sources = train_sources if isinstance(train_sources, (list, tuple)) else [train_sources]
        val_sources = data_yaml["val"].get("yolo_data", [])
        val_sources = val_sources if isinstance(val_sources, (list, tuple)) else [val_sources]
        train_data = [check_det_dataset(source) for source in train_sources]
        val_data = [check_det_dataset(source) for source in val_sources]
        assert val_data, "validation yolo dataset not found"

        grounding_data = data_yaml["train"].get("grounding_data") or []
        grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
        assert train_data or grounding_data, "training dataset not found"
        for g in grounding_data:
            assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            for k in {"img_path", "json_file"}:
                path = Path(g[k])
                if not path.exists() and not path.is_absolute():
                    g[k] = str((DATASETS_DIR / path).resolve())

        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            for d in [*train_data, *val_data]:
                d["names"] = {0: "object"}
                d["nc"] = 1

        self.validation_sets = []
        used_names = set()
        for i, (source, d) in enumerate(zip(val_sources, val_data)):
            if d.get("minival") is not None:
                d["minival"] = str(d["path"] / d["minival"])
            split = "minival" if "lvis" in str(d.get("val", "")) and d.get("minival") else "val"
            name = Path(str(source)).stem or f"val{i + 1}"
            if name in used_names:
                name = f"{name}-{i + 1}"
            used_names.add(name)
            self.validation_sets.append({"name": name, "source": source, "split": split, "path": d[split], "data": d})

        primary_val = self.validation_sets[0]
        max_text_samples = (
            1 if self.args.single_cls else 80 if grounding_data else min(max(d["nc"] for d in train_data), 80)
        )
        final_data = {
            "train": [d["train"] for d in train_data] + grounding_data,
            "val": primary_val["path"],
            "nc": primary_val["data"]["nc"],
            "names": primary_val["data"]["names"],
            "path": primary_val["data"]["path"],
            "channels": primary_val["data"]["channels"],
            "max_text_samples": max_text_samples,
        }
        self.data = final_data
        self.training_data = {}
        for d in train_data:
            self.training_data[d["train"]] = d
        return final_data

    def _build_train_pipeline(self):
        """Build the training loader and one validation loader per validation dataset."""
        super()._build_train_pipeline()
        self.test_loaders = [self.test_loader]
        if getattr(self, "validator", None) is not None:  # refresh after automatic OOM batch-size reduction
            self.validator.dataloader = self.test_loader
        if len(self.validation_sets) == 1:
            return

        batch_size = self.batch_size // max(self.world_size, 1)
        val_batch = batch_size if self.args.task in {"obb", "semantic"} else batch_size * 2
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        for val_set in self.validation_sets[1:]:
            with torch_distributed_zero_first(LOCAL_RANK):
                dataset = build_yolo_dataset(
                    self.args,
                    val_set["path"],
                    val_batch,
                    val_set["data"],
                    mode="val",
                    rect=False,
                    stride=gs,
                )
            self.test_loaders.append(
                build_dataloader(
                    dataset,
                    batch=val_batch,
                    workers=self.args.workers * 2,
                    shuffle=False,
                    rank=LOCAL_RANK,
                )
            )

    def validate(self):
        """Validate each dataset with its own vocabulary and average their fitness scores."""
        if len(self.validation_sets) == 1:
            return super().validate()
        if self.ema and self.world_size > 1:
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)

        original_data = self.data
        original_loader = self.validator.dataloader
        original_args = (self.validator.args.data, self.validator.args.split)
        metrics, fitness = {}, []
        try:
            # Run the primary dataset last so checkpoints retain its vocabulary and metric names.
            for i in [*range(1, len(self.validation_sets)), 0]:
                val_set = self.validation_sets[i]
                self.data = val_set["data"]
                self.validator.dataloader = self.test_loaders[i]
                self.validator.args.data = val_set["source"]
                self.validator.args.split = val_set["split"]
                result = self.validator(self)
                if result is None:
                    continue
                fitness.append(result.pop("fitness", -self.loss.detach().cpu().numpy()))
                if i == 0:
                    metrics.update(result)
                else:
                    metrics.update({f"{val_set['name']}/{key}": value for key, value in result.items()})
        finally:
            self.data = original_data
            self.validator.dataloader = original_loader
            self.validator.args.data, self.validator.args.split = original_args

        if not fitness:
            return None, None
        mean_fitness = sum(fitness) / len(fitness)
        if not self.best_fitness or self.best_fitness < mean_fitness:
            self.best_fitness = mean_fitness
        return metrics, mean_fitness

    def plot_training_labels(self):
        """Skip label plotting for YOLO-World training."""
        pass

    def final_eval(self):
        """Validate the best checkpoint independently on every validation dataset."""
        primary = self.validation_sets[0]
        self.validator.dataloader = self.test_loaders[0]
        self.validator.args.data = primary["source"]
        self.validator.args.split = primary["split"]
        super().final_eval()

        model = self.best if self.best.exists() else None
        if model:
            for i, val_set in enumerate(self.validation_sets[1:], 1):
                self.validator.dataloader = self.test_loaders[i]
                self.validator.args.data = val_set["source"]
                self.validator.args.split = val_set["split"]
                metrics = self.validator(model=model)
                metrics.pop("fitness", None)
                self.metrics.update({f"{val_set['name']}/{key}": value for key, value in metrics.items()})

        self.validator.dataloader = self.test_loaders[0]
        self.validator.args.data = primary["source"]
        self.validator.args.split = primary["split"]
        return self.metrics
