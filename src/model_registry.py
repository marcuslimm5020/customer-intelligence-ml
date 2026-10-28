"""Promote or roll back local model artifacts using a small JSON registry."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.train import ARTIFACTS


REGISTRY_PATH = ARTIFACTS / "model_registry.json"
MODEL_FILENAME = "purchase_propensity_pipeline.joblib"


def read_registry(path: Path) -> dict:
    if not path.exists():
        return {"active": None, "history": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry.get("history"), list):
        raise ValueError(f"Invalid model registry format: {path}")
    return registry


def write_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(model_dir: Path, registry_path: Path = REGISTRY_PATH) -> Path:
    model_path = (model_dir / MODEL_FILENAME).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not (model_dir / "model_metrics.csv").is_file():
        raise FileNotFoundError(f"Candidate evaluation report not found in {model_dir}")
    if not (model_dir / "selected_model.txt").is_file():
        raise FileNotFoundError(f"Candidate selection record not found in {model_dir}")
    if not (model_dir / "holdout_calibration.csv").is_file():
        raise FileNotFoundError(f"Holdout evaluation report not found in {model_dir}; run src.evaluate first")
    manifest_path = model_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model provenance manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(model_path)
    if manifest.get("model_sha256") != actual_hash:
        raise ValueError("Model artifact checksum does not match its provenance manifest")
    registry = read_registry(registry_path)
    prior = registry.get("active")
    if prior:
        registry["history"].append(prior)
    elif (ARTIFACTS / MODEL_FILENAME).is_file():
        registry["history"].append(
            {
                "model_path": str((ARTIFACTS / MODEL_FILENAME).resolve()),
                "model_sha256": sha256_file(ARTIFACTS / MODEL_FILENAME),
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "source": "pre-registry baseline",
            }
        )
    registry["active"] = {
        "model_path": str(model_path),
        "model_sha256": actual_hash,
        "selected_model": manifest.get("selected_model"),
        "manifest_path": str(manifest_path.resolve()),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_registry(registry_path, registry)
    return model_path


def rollback(registry_path: Path = REGISTRY_PATH) -> Path:
    registry = read_registry(registry_path)
    history = registry["history"]
    if not history:
        raise ValueError("No previous promoted model is available for rollback")
    previous = history.pop()
    model_path = Path(previous["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Previous model artifact is missing: {model_path}")
    if previous.get("model_sha256") and sha256_file(model_path) != previous["model_sha256"]:
        raise ValueError("Previous model checksum has changed; refusing to roll back to a modified artifact")
    current = registry.get("active")
    if current:
        registry["history"].append(current)
    registry["active"] = previous
    write_registry(registry_path, registry)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--promote", type=Path, help="Candidate model output directory")
    action.add_argument("--rollback", action="store_true", help="Restore the previous promoted model")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    try:
        active_model = (
            promote(args.promote, args.registry)
            if args.promote
            else rollback(args.registry)
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Active model: {active_model}")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()
