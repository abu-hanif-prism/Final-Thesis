"""Smoke-test reusable losses and metrics with fake data."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    confusion_matrix_np,
    multitask_metrics,
    regression_metrics,
    summarize_metrics,
)
from src.training.losses import (  # noqa: E402
    ClassificationLoss,
    MultiTaskLoss,
    RegressionLoss,
    compute_loss,
    get_loss_function,
)


def main() -> None:
    """Run loss, metric, and backprop checks."""
    torch.manual_seed(42)
    batch_size = 8
    batch = {
        "change_ratio": torch.rand(batch_size),
        "change_class_id": torch.randint(0, 3, (batch_size,)),
    }
    regression_pred = torch.rand(batch_size)
    classification_logits = torch.randn(batch_size, 3)
    multitask_outputs = {
        "change_ratio_pred": torch.rand(batch_size),
        "change_class_logits": torch.randn(batch_size, 3),
    }

    print("Regression losses:")
    for loss_type in ["mse", "mae", "huber", "smooth_l1"]:
        loss_fn = RegressionLoss(loss_type=loss_type)
        loss = loss_fn(regression_pred, batch["change_ratio"])
        print(f"  {loss_type}: {loss.item():.6f}, finite={torch.isfinite(loss).item()}")

    print("\nClassification losses:")
    cls_loss = ClassificationLoss()
    weighted_cls_loss = ClassificationLoss(class_weights=[1.0, 1.5, 0.75])
    cls_value = cls_loss(classification_logits, batch["change_class_id"])
    weighted_cls_value = weighted_cls_loss(classification_logits, batch["change_class_id"])
    print(f"  unweighted: {cls_value.item():.6f}, finite={torch.isfinite(cls_value).item()}")
    print(f"  weighted: {weighted_cls_value.item():.6f}, finite={torch.isfinite(weighted_cls_value).item()}")

    print("\nMultitask loss:")
    multitask_loss_fn = MultiTaskLoss(regression_weight=1.0, classification_weight=1.0)
    multitask_loss = multitask_loss_fn(multitask_outputs, batch)
    for key, value in multitask_loss.items():
        print(f"  {key}: {value.item():.6f}, finite={torch.isfinite(value).item()}")

    print("\ncompute_loss helper:")
    for output_mode, outputs, loss_fn in [
        ("regression", regression_pred, get_loss_function("regression", loss_type="mse")),
        ("classification", classification_logits, get_loss_function("classification")),
        ("multitask", multitask_outputs, get_loss_function("multitask")),
    ]:
        loss_dict = compute_loss(outputs, batch, output_mode, loss_fn)
        printable = {key: round(value.item(), 6) for key, value in loss_dict.items()}
        print(f"  {output_mode}: {printable}")

    print("\nMetrics:")
    reg_metrics = regression_metrics(regression_pred, batch["change_ratio"])
    cls_metrics = classification_metrics(classification_logits, batch["change_class_id"])
    mt_metrics = multitask_metrics(multitask_outputs, batch)
    print(f"  regression_metrics: {reg_metrics}")
    print(f"  classification_metrics: {cls_metrics}")
    print(f"  multitask_metrics: {mt_metrics}")
    print(f"  regression summary: {summarize_metrics(reg_metrics)}")
    print(f"  classification summary: {summarize_metrics(cls_metrics)}")

    pred_classes = classification_logits.argmax(dim=1)
    confusion = confusion_matrix_np(pred_classes, batch["change_class_id"], num_classes=3)
    print("\nConfusion matrix:")
    print(confusion)
    print(f"Confusion matrix shape: {confusion.shape}")

    print("\nBackprop checks:")
    run_backprop_checks(batch_size, batch)
    print("Backprop check passed")


def run_backprop_checks(batch_size: int, batch: dict[str, torch.Tensor]) -> None:
    """Verify gradients flow through all loss modes."""
    reg_pred = torch.rand(batch_size, requires_grad=True)
    reg_loss = RegressionLoss("mse")(reg_pred, batch["change_ratio"])
    reg_loss.backward()
    assert reg_pred.grad is not None
    assert torch.isfinite(reg_pred.grad).all()

    cls_logits = torch.randn(batch_size, 3, requires_grad=True)
    cls_loss = ClassificationLoss()(cls_logits, batch["change_class_id"])
    cls_loss.backward()
    assert cls_logits.grad is not None
    assert torch.isfinite(cls_logits.grad).all()

    mt_reg_pred = torch.rand(batch_size, requires_grad=True)
    mt_cls_logits = torch.randn(batch_size, 3, requires_grad=True)
    mt_loss = MultiTaskLoss()(
        {"change_ratio_pred": mt_reg_pred, "change_class_logits": mt_cls_logits},
        batch,
    )["total_loss"]
    mt_loss.backward()
    assert mt_reg_pred.grad is not None
    assert mt_cls_logits.grad is not None
    assert torch.isfinite(mt_reg_pred.grad).all()
    assert torch.isfinite(mt_cls_logits.grad).all()


if __name__ == "__main__":
    main()
