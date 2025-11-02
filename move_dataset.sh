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

# Summary
echo "---- Summary ----"
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