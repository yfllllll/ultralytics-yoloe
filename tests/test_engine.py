# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL, SOURCE, TASK_MODEL_DATA
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, detect, obb, pose, segment, semantic
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.torch_utils import unwrap_model


def test_func(*args, **kwargs):
    """Test function used as a callback stub to verify callback registration."""
    print("callback test passed")


def test_export(monkeypatch, tmp_path):
    """Test model exporting functionality by adding a callback and verifying its execution."""
    monkeypatch.chdir(tmp_path)
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "on_export_start callback not registered"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


@pytest.mark.parametrize(
    "trainer_cls,validator_cls,predictor_cls,data,model,weights",
    [
        (
            detect.DetectionTrainer,
            detect.DetectionValidator,
            detect.DetectionPredictor,
            "coco8.yaml",
            "yolo26n.yaml",
            MODEL,
        ),
        (
            segment.SegmentationTrainer,
            segment.SegmentationValidator,
            segment.SegmentationPredictor,
            "coco8-seg.yaml",
            "yolo26n-seg.yaml",
            WEIGHTS_DIR / "yolo26n-seg.pt",
        ),
        (
            classify.ClassificationTrainer,
            classify.ClassificationValidator,
            classify.ClassificationPredictor,
            "imagenet10",
            "yolo26n-cls.yaml",
            None,
        ),
        (obb.OBBTrainer, obb.OBBValidator, obb.OBBPredictor, "dota8.yaml", "yolo26n-obb.yaml", None),
        (pose.PoseTrainer, pose.PoseValidator, pose.PosePredictor, "coco8-pose.yaml", "yolo26n-pose.yaml", None),
        (
            semantic.SemanticSegmentationTrainer,
            semantic.SemanticSegmentationValidator,
            semantic.SemanticSegmentationPredictor,
            "cityscapes8.yaml",
            "yolo26n-sem.yaml",
            None,
        ),
    ],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_task(trainer_cls, validator_cls, predictor_cls, data, model, weights):
    """Test YOLO training, validation, and prediction for various tasks."""
    overrides = {
        "data": data,
        "model": model,
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }

    # Trainer
    trainer = trainer_cls(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "on_train_start callback not registered"
    trainer.train()

    # Validator
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data
    cfg.imgsz = 32
    val = validator_cls(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "on_val_start callback not registered"
    val(model=trainer.best)

    # Predictor
    pred = predictor_cls(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "on_predict_start callback not registered"

    # Determine model path for prediction
    model_path = weights if weights else trainer.best
    if model == "yolo26n.yaml":  # only for detection
        # Confirm there is no issue with sys.argv being empty
        with mock.patch.object(sys, "argv", []):
            result = pred(source=ASSETS, model=model_path)
            assert len(result) > 0, f"Predictor returned no results for {model}"
    else:
        result = pred(source=ASSETS, model=model_path)
        assert len(result) > 0, f"Predictor returned no results for {model}"

    # Test resume functionality
    with pytest.raises(AssertionError):
        trainer_cls(overrides={**overrides, "resume": trainer.last}).train()


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_resume_incomplete(task, weight, data, tmp_path):
    """Test training resumes from an incomplete checkpoint."""
    train_args = {
        "data": data,
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": task,
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO(weight)
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**train_args)
    last_path = model.trainer.last
    _, ckpt = load_checkpoint(last_path)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # Resume training using the checkpoint
    resume_model = YOLO(last_path)
    resume_model.train(resume=True, **train_args)
    assert resume_model.trainer.start_epoch == resume_model.trainer.epoch == 1, "resume test failed"


def test_distill_resume(tmp_path: Path):
    """Test knowledge distillation resumes from an incomplete checkpoint."""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "distill_model": WEIGHTS_DIR / "yolo26s.pt",
        "imgsz": 32,
        "multi_scale": 0.5,  # vary per-batch image size to exercise dynamic distillation score splitting
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill",
        "exist_ok": True,
    }

    # Train for one epoch then interrupt to produce a resumable checkpoint
    trainer = detect.DetectionTrainer(overrides=overrides)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"
    assert isinstance(ckpt["ema"], DistillationModel), "distillation EMA wraps the student model"
    assert ckpt["ema"].teacher_model is None, "teacher should be stripped from the EMA/checkpoint"
    assert ckpt["ema"].projector is not None, "the distillation projector should be persisted in the EMA checkpoint"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationModel), "resume should rebuild the DistillationModel"
    assert model.teacher_model is not None, "resume should rebuild the teacher from the distill_model path"
    assert trainer.start_epoch == trainer.epoch == 1, "resume test failed"


def test_distill_grayscale(tmp_path: Path):
    """Test knowledge distillation on a single-channel dataset (https://github.com/ultralytics/ultralytics/issues/25066)."""
    teacher = DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher}, teacher_path)
    student = DetectionModel("yolo26n.yaml", ch=1, nc=80, verbose=False)
    student.args = SimpleNamespace(imgsz=32, dis=1.0)
    model = DistillationModel(teacher_model=teacher_path, student_model=student)
    assert isinstance(model, DistillationModel)
    assert model.teacher_model.yaml["channels"] == 1


@pytest.mark.parametrize(
    "ckpt",
    [
        {"model": OrderedDict([("a", torch.zeros(1))])},  # state_dict saved under the "model" key
        {"model": {"a": torch.zeros(1)}},  # plain-dict "model" value
        OrderedDict([("a", torch.zeros(1))]),  # bare state_dict, no "model" key
    ],
)
def test_load_checkpoint_state_dict_rejected(ckpt, tmp_path):
    """Test a state_dict checkpoint raises a clear TypeError instead of a cryptic AttributeError/KeyError."""
    weight = tmp_path / "bad.pt"
    torch.save(ckpt, weight)
    with pytest.raises(TypeError, match="supported Ultralytics checkpoint format"):
        load_checkpoint(weight)


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_checkpoint_fp16_overflow():
    """Test a finite model whose weights overflow fp16 is still checkpointed (clamped) instead of skipped."""

    def inflate_ema(trainer):
        """Push an EMA weight above the fp16 max (65504) so its fp16 snapshot would otherwise become Inf."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = 1.0e5

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", inflate_ema)
    trainer.train()
    assert trainer.last.exists(), "checkpoint not saved for a finite model with fp16-overflowing weights"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )
    # Validation must leave the live EMA fp32 and unchanged; checkpoint serialization may clamp its fp16 copy.
    ema_param = next(iter(trainer.ema.ema.parameters()))
    assert ema_param.dtype == torch.float32 and torch.isfinite(ema_param).all() and ema_param.flatten()[0] == 1.0e5, (
        "validation corrupted the live EMA"
    )


def test_checkpoint_nonfinite_ema_resync():
    """Test a non-finite EMA on a finite model is resynced (not skipped) so the run still produces a checkpoint."""

    def poison_ema(trainer):
        """Make the live fp32 EMA genuinely non-finite while the model stays finite (sticky-NaN on a finite-loss run)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when the EMA went non-finite on a finite model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


def test_checkpoint_nonfinite_ema_and_model_sanitized():
    """Test a tensor non-finite in both EMA and model is sanitized (not skipped) so the run still produces a checkpoint."""

    def poison_ema_and_model(trainer):
        """Force the first parameter non-finite in both the live EMA and the model (finite-loss sticky-NaN)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")
            next(iter(unwrap_model(trainer.model).parameters())).data.flatten()[0] = float("nan")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema_and_model)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when a tensor went non-finite in both EMA and model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


@pytest.mark.parametrize(
    "kwargs,uses_weights",
    [({}, True), ({"pretrained": True}, True), ({"pretrained": False}, False), ({"pretrained": MODEL}, True)],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_reuses_loaded_checkpoint_model(monkeypatch, kwargs, uses_weights):
    """Test training reuses loaded checkpoint config while respecting the pretrained argument."""
    model = YOLO("yolo26n.yaml")
    model.ckpt = {"checkpoint": True}
    model.ckpt_path = "/tmp/fake.pt"
    model.overrides["model"] = "ul://glenn-jocher/m2/exp-14"
    model.overrides["pretrained"] = False
    original_model = model.model
    captured = {}

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = None
            self.validator = SimpleNamespace(metrics=None)
            self.best = MODEL.parent / "nonexistent-best.pt"
            self.last = MODEL
            captured["trainer"] = self

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["cfg"] = cfg
            captured["weights"] = weights
            return original_model

        def train(self):
            return None

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    monkeypatch.setattr(model, "_smart_load", lambda key: FakeTrainer)
    monkeypatch.setattr(
        "ultralytics.engine.model.load_checkpoint",
        lambda path: (original_model, {"checkpoint": True}),
    )

    model.train(data="coco8.yaml", epochs=1, **kwargs)

    assert captured["trainer"].model is original_model, "Trainer model does not match original"
    assert captured["cfg"] == original_model.yaml, f"Config mismatch: {captured['cfg']} != {original_model.yaml}"
    assert captured["weights"] is (original_model if uses_weights else None), "Unexpected weights loaded"


def test_train_multi_custom_trainer_metrics_and_failure_keys(monkeypatch, tmp_path):
    """Test custom multi-dataset runs keep memory metrics and unique failure keys."""
    model = YOLO(MODEL)
    calls = 0

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            pass

        def get_model(self, cfg=None, weights=None, verbose=True):
            return model.model

        def train(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("failed repeated dataset")
            self.validator = SimpleNamespace(metrics=SimpleNamespace(results_dict={"fitness": 1.0}))

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    results = model.train(
        data=["coco8.yaml", "coco8.yaml"],
        project=tmp_path,
        plots=False,
        save=False,
        trainer=FakeTrainer,
    )

    assert model.trainer.trainer is FakeTrainer
    assert results == {"coco8": {"fitness": 1.0}, "coco8-2": None}


@pytest.mark.parametrize("pretrained,uses_weights", [(True, True), (False, False), (MODEL, True)])
def test_setup_model_respects_pretrained_arg_for_pt_models(monkeypatch, pretrained, uses_weights):
    """Test .pt models use checkpoint config while respecting the pretrained argument."""
    captured = {}
    checkpoint_model = SimpleNamespace(yaml={"nc": 80})
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n.pt"
    trainer.args = SimpleNamespace(pretrained=pretrained)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(
        "ultralytics.engine.trainer.load_checkpoint", lambda path: (checkpoint_model, {"checkpoint": True})
    )

    trainer.setup_model()

    assert captured["cfg"] == checkpoint_model.yaml, "Checkpoint config was not used"
    assert captured["weights"] is (checkpoint_model if uses_weights else None), "Unexpected weights loaded"


def test_world_trainer_pads_per_dataset_texts_without_mixing_boundaries():
    """Pad heterogeneous local vocabularies while preserving each image's text features and supervision mask."""
    from ultralytics.models.yolo.world.train import WorldTrainer

    trainer = object.__new__(WorldTrainer)
    trainer.args = SimpleNamespace(multi_scale=0.0)
    trainer.device = torch.device("cpu")
    trainer.model = SimpleNamespace(model=[SimpleNamespace(nc=4)])
    trainer.text_embeddings = {name: torch.full((3,), value) for value, name in enumerate("abcdef", 1)}
    batch = {
        "img": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        "texts": (["a", "b"], ["c", "d", "e", "f"]),
    }

    batch = trainer.preprocess_batch(batch)

    assert batch["txt_feats"].shape == (2, 4, 3)
    assert torch.equal(batch["txt_feats"][0, :2], torch.stack([trainer.text_embeddings[x] for x in "ab"]))
    assert torch.equal(batch["txt_feats"][1], torch.stack([trainer.text_embeddings[x] for x in "cdef"]))
    assert not batch["txt_feats"][0, 2:].any()
    assert torch.equal(batch["text_mask"], torch.tensor([[True, True, False, False], [True, True, True, True]]))


def test_yoloe_model_preserves_configured_prompt_capacity():
    """Keep the configured prompt count during the dummy forward used to initialize model strides."""
    from ultralytics.nn.tasks import YOLOEModel

    model = YOLOEModel("ultralytics/cfg/models/11/yoloe-11.yaml", nc=5, verbose=False)

    assert model.model[-1].nc == 5


def test_chinese_clip_is_frozen_and_normalizes_features(monkeypatch):
    """Keep Chinese-CLIP outside optimization while returning normalized 512-dimensional prompt features."""
    import transformers

    from ultralytics.nn.text_model import ChineseCLIP

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            shape = (len(texts), 4)
            return {
                "input_ids": torch.ones(shape, dtype=torch.long),
                "attention_mask": torch.ones(shape, dtype=torch.long),
            }

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def get_text_features(self, input_ids, attention_mask):
            return self.weight * input_ids[:, :1].repeat(1, 512)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(transformers.ChineseCLIPModel, "from_pretrained", lambda *args, **kwargs: Encoder())

    model = ChineseCLIP("test/chinese-clip", torch.device("cpu"))
    features = model.encode_text(model.tokenize(["安全帽", "反光背心"]))

    assert features.shape == (2, 512)
    assert torch.allclose(features.norm(dim=-1), torch.ones(2))
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not features.requires_grad


def test_yoloe_model_switches_text_encoder_without_loading_it():
    """Persist the selected encoder through trainer reconstruction without attaching its weights to YOLOE."""
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.train import YOLOETrainer

    wrapper = YOLOE("ultralytics/cfg/models/11/yoloe-11.yaml")
    variant = "chineseclip:OFA-Sys/chinese-clip-vit-base-patch16"

    wrapper.set_text_model(variant)
    model = wrapper.model

    assert model.text_model == variant
    assert model.yaml["text_model"] == variant
    assert model.clip_model is None
    assert wrapper.overrides["text_model"] == variant

    trainer = object.__new__(YOLOETrainer)
    trainer.data = {"channels": 3, "max_text_samples": 5, "nc": 5}
    trainer.args = SimpleNamespace(text_model=variant)
    rebuilt = trainer.get_model(cfg=model.yaml, verbose=False)

    assert rebuilt.text_model == variant
    assert getattr(rebuilt, "clip_model", None) is None


def test_world_trainer_applies_shared_text_limit_to_yolo_datasets(monkeypatch):
    """Apply the training/model text capacity to every YOLO child dataset without merging their vocabularies."""
    from ultralytics.models.yolo.world import train_world

    max_samples = []

    def build_dataset(*args, **kwargs):
        max_samples.append(kwargs["max_samples"])
        return torch.utils.data.TensorDataset(torch.zeros(1))

    trainer = object.__new__(train_world.WorldTrainerFromScratch)
    trainer.args = SimpleNamespace()
    trainer.data = {"max_text_samples": 5}
    trainer.model = SimpleNamespace(stride=torch.tensor([32]))
    trainer.training_data = {"small": {}, "large": {}}
    trainer.set_text_embeddings = lambda datasets, batch: None
    monkeypatch.setattr(train_world, "build_yolo_dataset", build_dataset)

    dataset = trainer.build_dataset(["small", "large"], batch=2)

    assert len(dataset.datasets) == 2
    assert max_samples == [5, 5]


@pytest.mark.parametrize("single_cls,expected_max", [(False, 5), (True, 1)])
def test_world_trainer_supports_multiple_validation_datasets(monkeypatch, single_cls, expected_max):
    """Derive prompt capacity from training data and retain independent validation metadata."""
    from ultralytics.models.yolo.world import train_world

    datasets = {
        "train-one.yaml": {"train": "train-one", "val": "unused", "nc": 1},
        "train-five.yaml": {"train": "train-five", "val": "unused", "nc": 5},
        "val-two.yaml": {"train": "unused", "val": "val-two", "nc": 2},
        "val-seven.yaml": {"train": "unused", "val": "val-seven", "nc": 7},
    }
    for data in datasets.values():
        data.update(names=dict(enumerate(f"class-{i}" for i in range(data["nc"]))), path=Path(), channels=3)

    trainer = object.__new__(train_world.WorldTrainerFromScratch)
    trainer.args = SimpleNamespace(
        data={
            "train": {"yolo_data": ["train-one.yaml", "train-five.yaml"]},
            "val": {"yolo_data": ["val-two.yaml", "val-seven.yaml"]},
        },
        single_cls=single_cls,
    )
    monkeypatch.setattr(train_world, "check_det_dataset", lambda source: datasets[source].copy())

    data = trainer.get_dataset()

    assert data["max_text_samples"] == expected_max
    assert data["val"] == "val-two"
    assert len(trainer.validation_sets) == 2
    assert [x["path"] for x in trainer.validation_sets] == ["val-two", "val-seven"]
    assert [x["data"]["nc"] for x in trainer.validation_sets] == ([1, 1] if single_cls else [2, 7])


def test_world_trainer_aggregates_multiple_validation_metrics():
    """Keep primary metric names, prefix secondary metrics, and average dataset fitness equally."""
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    class Validator:
        def __init__(self):
            self.args = SimpleNamespace(data="original", split="val")
            self.dataloader = "primary-loader"

        def __call__(self, trainer):
            return {"metrics/mAP50-95(B)": trainer.data["score"], "fitness": trainer.data["score"]}

    trainer = object.__new__(WorldTrainerFromScratch)
    trainer.data = {"original": True}
    trainer.validation_sets = [
        {"name": "primary", "source": "primary.yaml", "split": "val", "data": {"score": 0.2}},
        {"name": "secondary", "source": "secondary.yaml", "split": "val", "data": {"score": 0.8}},
    ]
    trainer.test_loaders = ["primary-loader", "secondary-loader"]
    trainer.validator = Validator()
    trainer.ema = None
    trainer.world_size = 1
    trainer.loss = torch.tensor(1.0)
    trainer.best_fitness = 0.0

    metrics, fitness = trainer.validate()

    assert metrics == {"metrics/mAP50-95(B)": 0.2, "secondary/metrics/mAP50-95(B)": 0.8}
    assert fitness == 0.5
    assert trainer.data == {"original": True}
    assert trainer.validator.dataloader == "primary-loader"


def test_detection_loss_ignores_masked_text_slots():
    """Ensure padded text slots cannot suppress classes absent from a child dataset's vocabulary."""
    from ultralytics.utils.loss import v8DetectionLoss

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nc = 4
            self.reg_max = 1
            self.stride = torch.tensor([8.0])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(box=1.0, cls=1.0, dfl=1.0)
            self.model = torch.nn.ModuleList([Head()])

    criterion = v8DetectionLoss(Model())
    preds = {
        "boxes": torch.zeros(2, 4, 1),
        "scores": torch.zeros(2, 4, 1),
        "feats": [torch.zeros(2, 1, 1, 1)],
    }
    batch = {
        "batch_idx": torch.empty(0),
        "cls": torch.empty(0),
        "bboxes": torch.empty(0, 4),
        "text_mask": torch.tensor([[True, True, False, False], [True, True, True, True]]),
    }
    base_loss = criterion.loss(preds, batch)[0]
    preds["scores"][0, 2:] = 20  # masked logits must not alter the classification loss

    assert torch.equal(criterion.loss(preds, batch)[0], base_loss)
