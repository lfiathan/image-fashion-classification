from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
	# src/utils/paths.py -> parents[0]=src/utils, [1]=src, [2]=project root
	return Path(__file__).resolve().parents[2]

def src_dir() -> Path:
	return repo_root() / "src"

def notebooks_dir() -> Path:
	return repo_root() / "notebooks"

def data_dir() -> Path:
	return repo_root() / "data"

__all__ = [
	"repo_root",
	"src_dir",
	"notebooks_dir",
	"data_dir",
]
