"""Command-line entry point for one tracked training experiment."""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from signet_experiments.dataset import create_data_loader
from signet_experiments.metrics import classification_metrics, select_threshold
from signet_experiments.model import SiameseNetwork
from signet_experiments.splits import (
    PAIR_COLUMNS,
    create_pairs,
    discover_person_ids,
    split_person_ids,
    write_csv,
)

DEFAULT_DATASET = "akashgundu/signature-verification-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one Siamese signature-verification run."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/signatures/extract"),
        help="Directory containing signer and <signer>_forg image folders.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download the Kaggle dataset ({DEFAULT_DATASET}) if needed.",
    )
    parser.add_argument("--tracking-uri", default="./mlruns")
    parser.add_argument(
        "--experiment-name",
        default="Signature Verification Experiments",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--image-height", type=int, default=155)
    parser.add_argument("--image-width", type=int, default=220)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_data_dir(data_dir: Path, download: bool) -> Path:
    if download:
        import kagglehub

        downloaded_dir = Path(kagglehub.dataset_download(DEFAULT_DATASET))
        extracted_dir = downloaded_dir / "extract"
        data_dir = extracted_dir if extracted_dir.is_dir() else downloaded_dir
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. Pass --data-dir or use --download."
        )
    return data_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: SiameseNetwork,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for image1, image2, labels in loader:
        image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(image1, image2), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        sample_count += labels.size(0)
    if not sample_count:
        raise ValueError("Training split has no batches.")
    return total_loss / sample_count


def evaluate(
    model: SiameseNetwork,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for image1, image2, labels in loader:
            image1, image2 = image1.to(device), image2.to(device)
            labels = labels.to(device)
            logits = model(image1, image2)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            sample_count += labels.size(0)
            all_labels.append(labels.cpu().numpy().reshape(-1))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    if not sample_count:
        raise ValueError("Evaluation split has no batches.")
    return (
        total_loss / sample_count,
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def log_plots(
    artifact_dir: Path,
    train_losses: list[float],
    validation_losses: list[float],
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    threshold: float,
    test_confusion: np.ndarray,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, validation_losses, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(artifact_dir / "training_curve.png", dpi=160)
    plt.close()

    thresholds = np.linspace(0.01, 0.99, 99)
    far_values = []
    frr_values = []
    for candidate in thresholds:
        metrics, _ = classification_metrics(
            validation_labels, validation_probabilities, float(candidate)
        )
        far_values.append(metrics["far"] * 100)
        frr_values.append(metrics["frr"] * 100)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far_values, label="False acceptance rate (forged)")
    plt.plot(thresholds, frr_values, label="False rejection rate (genuine)")
    plt.axvline(threshold, linestyle="--", label=f"Selected = {threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate (%)")
    plt.title("Validation FAR/FRR by threshold")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(artifact_dir / "far_frr_curve.png", dpi=160)
    plt.close()

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(test_confusion, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Genuine", "Forged"])
    axis.set_yticks([0, 1], labels=["Genuine", "Forged"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Test confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(test_confusion[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(artifact_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Batch size, max epochs, and patience must be positive.")
    if args.image_height < 1 or args.image_width < 1 or args.embedding_dim < 1:
        raise ValueError("Image dimensions and embedding dimension must be positive.")
    if args.num_workers < 0 or not 0 <= args.dropout < 1:
        raise ValueError("num-workers must be non-negative and dropout in [0, 1).")

    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir, args.download)
    person_ids = discover_person_ids(data_dir)
    train_ids, validation_ids, test_ids = split_person_ids(
        person_ids,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
    )
    split_pairs = {
        "train": create_pairs(data_dir, train_ids),
        "validation": create_pairs(data_dir, validation_ids),
        "test": create_pairs(data_dir, test_ids),
    }
    pair_frames = {
        name: pd.DataFrame.from_records(pairs, columns=PAIR_COLUMNS)
        for name, pairs in split_pairs.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loaders = {
        "train": create_data_loader(
            pair_frames["train"], data_dir, args.image_height, args.image_width,
            args.batch_size, True, args.num_workers, args.seed, pin_memory,
        ),
        "validation": create_data_loader(
            pair_frames["validation"], data_dir, args.image_height, args.image_width,
            args.batch_size, False, args.num_workers, args.seed, pin_memory,
        ),
        "test": create_data_loader(
            pair_frames["test"], data_dir, args.image_height, args.image_width,
            args.batch_size, False, args.num_workers, args.seed, pin_memory,
        ),
    }

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or f"siamese-seed-{args.seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "SiameseNetwork",
                "image_height": args.image_height,
                "image_width": args.image_width,
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "optimizer": "Adam",
                "loss_function": "BCEWithLogitsLoss",
                "max_epochs": args.max_epochs,
                "early_stopping_patience": args.patience,
                "random_seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "test_fraction": args.test_fraction,
                "device": str(device),
            }
        )
        mlflow.set_tag("dataset_directory", str(data_dir))
        mlflow.log_metrics(
            {
                "training_persons": len(train_ids),
                "validation_persons": len(validation_ids),
                "test_persons": len(test_ids),
                "training_pairs": len(split_pairs["train"]),
                "validation_pairs": len(split_pairs["validation"]),
                "test_pairs": len(split_pairs["test"]),
            }
        )

        model = SiameseNetwork(args.embedding_dim, args.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []

        with tempfile.TemporaryDirectory(prefix="signet-experiment-") as temp_dir:
            artifact_dir = Path(temp_dir)
            checkpoint_path = artifact_dir / "best_model.pth"
            for epoch in range(args.max_epochs):
                train_loss = train_one_epoch(
                    model, loaders["train"], optimizer, criterion, device
                )
                validation_loss, validation_labels, validation_probabilities = evaluate(
                    model, loaders["validation"], criterion, device
                )
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    },
                    step=epoch + 1,
                )
                print(
                    f"Epoch {epoch + 1}/{args.max_epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f}"
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print("Early stopping.")
                        break

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            _, validation_labels, validation_probabilities = evaluate(
                model, loaders["validation"], criterion, device
            )
            threshold, validation_metrics = select_threshold(
                validation_labels, validation_probabilities
            )
            test_loss, test_labels, test_probabilities = evaluate(
                model, loaders["test"], criterion, device
            )
            test_metrics, test_confusion = classification_metrics(
                test_labels, test_probabilities, threshold
            )
            mlflow.log_metrics(
                {
                    "best_validation_loss": best_validation_loss,
                    "validation_threshold": threshold,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    "test_loss": test_loss,
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )
            mlflow.log_param("threshold_selection_method", "max_validation_f1")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "embedding_dim": args.embedding_dim,
                        "dropout": args.dropout,
                        "image_height": args.image_height,
                        "image_width": args.image_width,
                    },
                    "threshold": threshold,
                },
                artifact_dir / "best_model.pth",
            )
            log_plots(
                artifact_dir,
                train_losses,
                validation_losses,
                validation_labels,
                validation_probabilities,
                threshold,
                test_confusion,
            )
            for split_name, ids in (
                ("train", train_ids),
                ("validation", validation_ids),
                ("test", test_ids),
            ):
                write_csv(
                    artifact_dir / f"{split_name}_persons.csv",
                    ("person_id",),
                    ((person_id,) for person_id in ids),
                )
                write_csv(
                    artifact_dir / f"{split_name}_pairs.csv",
                    PAIR_COLUMNS,
                    split_pairs[split_name],
                )
            mlflow.log_artifacts(str(artifact_dir))

        print(
            f"Test F1={test_metrics['f1_forged']:.4f} | "
            f"accuracy={test_metrics['accuracy']:.4f} | "
            f"FAR={test_metrics['far']:.4f} | "
            f"FRR={test_metrics['frr']:.4f} | threshold={threshold:.2f}"
        )
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
