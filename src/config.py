from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

from src.utils.paths import repo_root


@dataclass(frozen=True)
class Config:
	"""Project-wide configuration with sensible defaults.

	Override via environment variables when needed:
	  - DATASET_CLASSIFY_DIR
	  - FASHION_MNIST_ZIP
	  - FASHION_MNIST_DIR
	  - YOLORUNS_DIR
	"""

	REPO_ROOT: Path = repo_root()

	DATASET_CLASSIFY_DIR: Path = Path(
		os.environ.get("DATASET_CLASSIFY_DIR", str(repo_root() / "dataset-classify"))
	)
	FASHION_MNIST_ZIP: Path = Path(
		os.environ.get("FASHION_MNIST_ZIP", str(repo_root() / "fashion-mnist.zip"))
	)
	FASHION_MNIST_DIR: Path = Path(
		os.environ.get("FASHION_MNIST_DIR", str(repo_root() / "fashion-mnist"))
	)
	YOLORUNS_DIR: Path = Path(
		os.environ.get("YOLORUNS_DIR", str(repo_root() / "runs-cls"))
	)
