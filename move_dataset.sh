#!/usr/bin/env bash
set -euo pipefail

# Run this from project root (or adjust BASE)
BASE="/Users/macm4/repositories/Machine Learning Model/modisch-model-cls/data"
DATASET_DIR="$BASE/dataset-fashion-modisch"
TRAIN_DIR="$DATASET_DIR/train"
VAL_DIR="$DATASET_DIR/val"

mkdir -p "$TRAIN_DIR" "$VAL_DIR"

# --- Clothes mapping (all moved into train/) ---
CLOTHES_BASE="$BASE/clothes-dataset-unzipped/Clothes_Dataset"

cat <<'MAP' | while IFS='|' read -r SRCNAME TGT; do
Blazer|blazer
Celana_Panjang|pants
Celana_Pendek|short
Gaun|dress
Hoodie|hoodie
Jaket|jacket
Jaket_Denim|jacket
Jaket_Olahraga|jacket
Jeans|pants
Kaos|t-shirt
Kemeja|shirt
Mantel|jacket
Polo|polo
Rok|skirt
Sweter|sweater
MAP
  SRCDIR="$CLOTHES_BASE/$SRCNAME"
  TGTDIR="$TRAIN_DIR/$TGT"
  if [ -d "$SRCDIR" ]; then
    mkdir -p "$TGTDIR"
    echo "Moving clothes: $SRCNAME -> $TGTDIR"
    rsync -av --remove-source-files --no-perms --no-owner --no-group "$SRCDIR"/ "$TGTDIR"/
    find "$SRCDIR" -type d -empty -delete || true
  else
    echo "Warning: clothes source not found: $SRCDIR"
  fi
done

# --- Shoes mapping (training -> train, validation -> val) ---
SHOES_BASE="$BASE/shoe-dataset-unzipped/shoeTypeClassifierDataset"

cat <<'MAP' | while IFS='|' read -r SRCNAME TGT; do
boots|boot
sneakers|sneaker
flip_flops|flip-flop
loafers|loafer
sandals|slipper
soccer_shoes|sneaker
MAP
  SRC_TRAIN="$SHOES_BASE/training/$SRCNAME"
  TGT_TRAIN="$TRAIN_DIR/$TGT"
  if [ -d "$SRC_TRAIN" ]; then
    mkdir -p "$TGT_TRAIN"
    echo "Moving shoes (train): $SRC_TRAIN -> $TGT_TRAIN"
    rsync -av --remove-source-files --no-perms --no-owner --no-group "$SRC_TRAIN"/ "$TGT_TRAIN"/
    find "$SRC_TRAIN" -type d -empty -delete || true
  else
    echo "Warning: shoe training source not found: $SRC_TRAIN"
  fi

  SRC_VAL="$SHOES_BASE/validation/$SRCNAME"
  TGT_VAL="$VAL_DIR/$TGT"
  if [ -d "$SRC_VAL" ]; then
    mkdir -p "$TGT_VAL"
    echo "Moving shoes (val): $SRC_VAL -> $TGT_VAL"
    rsync -av --remove-source-files --no-perms --no-owner --no-group "$SRC_VAL"/ "$TGT_VAL"/
    find "$SRC_VAL" -type d -empty -delete || true
  else
    echo "Warning: shoe validation source not found: $SRC_VAL"
  fi
done

# --- Ensure 80/20 split: if a class lacks val images, move 20% from train -> val (deterministic seed) ---
export TRAIN_DIR VAL_DIR

python3 - "$TRAIN_DIR" "$VAL_DIR" <<'PY'
import os, random, pathlib, sys
random.seed(42)
ALLOWED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

train_dir = pathlib.Path(sys.argv[1])
val_dir = pathlib.Path(sys.argv[2])

# gather all class names present in either split
classes = set()
if train_dir.exists():
    classes |= {p.name for p in train_dir.iterdir() if p.is_dir()}
if val_dir.exists():
    classes |= {p.name for p in val_dir.iterdir() if p.is_dir()}

for cls in sorted(classes):
    tdir = train_dir / cls
    vdir = val_dir / cls
    tdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)

    train_imgs = [p for p in tdir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]
    val_imgs = [p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]

    total = len(train_imgs) + len(val_imgs)
    if total == 0:
        continue

    desired_val = int(total * 0.2)  # 20% to val
    need = desired_val - len(val_imgs)
    if need <= 0:
        continue

    # shuffle deterministically and move files from train -> val
    random.shuffle(train_imgs)
    to_move = train_imgs[:need]
    for p in to_move:
        dst = vdir / p.name
        i = 1
        while dst.exists():
            dst = vdir / f"{p.stem}_{i}{p.suffix}"
            i += 1
        p.rename(dst)

# print a brief summary
print("80/20 split enforced. Current counts per class (train / val):")
for cls in sorted(classes):
    tcount = len([p for p in (train_dir/cls).iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]) if (train_dir/cls).exists() else 0
    vcount = len([p for p in (val_dir/cls).iterdir() if p.is_file() and p.suffix.lower() in ALLOWED]) if (val_dir/cls).exists() else 0
    print(f" - {cls}: {tcount} / {vcount}")
PY

# Summary
echo "---- Final Summary ----"
echo "Train counts per class:"
for d in "$TRAIN_DIR"/*; do
  [ -d "$d" ] || continue
  echo "$(basename "$d"): $(find "$d" -type f | wc -l)"
done

echo "Val counts per class:"
for d in "$VAL_DIR"/*; do
  [ -d "$d" ] || continue
  echo "$(basename "$d"): $(find "$d" -type f | wc -l)"
done

echo "Done."