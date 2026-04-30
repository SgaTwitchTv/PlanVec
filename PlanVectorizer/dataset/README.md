# Annotation Classifier Dataset

This folder stores the labeled crop images for the first AI-assisted
annotation-vs-structure classifier.

Recommended workflow:

1. Run `python PlanVectorizer.py` on different floor images.
2. Open `output/debug/annotation_candidates/`.
3. Copy each exported crop PNG into exactly one class folder inside
   `annotation_classifier/`.
4. Put uncertain crops into `holdout_uncertain/` instead of forcing a bad label.
5. Train the classifier with:

```powershell
python tools/train_annotation_classifier.py dataset\annotation_classifier
```

The trained model will be saved to:

`models/annotation_classifier.npz`

Class rules:

- `annotation_room_label`: room numbers or number clusters like `202`, `203`,
  `202 20`, including blurry/distorted variants
- `annotation_callout`: small circular/oval indicator marks near labels
- `annotation_other`: other removable non-structural annotation remnants
- `structure_door`: door swings, door arcs, and small door-frame fragments that must stay
- `structure_square_pillar`: small square pillar-like structural elements
- `structure_rectangular_pillar`: small rectangular pillar-like structural elements
- `structure_other`: other true small structural objects that should stay

Notes:

- Any class folder that is still empty is skipped automatically during training.
- Use `structure_door` for the 90-degree door arc plus nearby jamb/leaf geometry when that
  is the main object in the crop.
