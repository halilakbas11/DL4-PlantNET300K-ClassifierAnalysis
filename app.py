import json
import io
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    densenet121, densenet169, densenet201,
    efficientnet_b0, efficientnet_b1, efficientnet_b4, efficientnet_b7,
    mobilenet_v3_small, mobilenet_v3_large,
)
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "models")
IDX_TO_SID_PATH  = os.path.join(BASE_DIR, "class_idx_to_species_id.json")
SID_TO_NAME_PATH = os.path.join(BASE_DIR, "plantnet300K_species_id_2_name.json")

N_CLASSES                = 1081
TOP_K                    = 5
CONFIDENCE_WARN_THRESHOLD = 15.0

# ── Architecture registry ─────────────────────────────────────────────────
# Maps the filename prefix (the part before "_weights") to its torchvision
# constructor.  To support a new architecture, add one line here and drop
# the matching .tar file into models/ — nothing else needs to change.
ARCH_REGISTRY = {
    "resnet18":           resnet18,
    "resnet34":           resnet34,
    "resnet50":           resnet50,
    "resnet101":          resnet101,
    "resnet152":          resnet152,
    "densenet121":        densenet121,
    "densenet169":        densenet169,
    "densenet201":        densenet201,
    "efficientnet_b0":    efficientnet_b0,
    "efficientnet_b1":    efficientnet_b1,
    "efficientnet_b4":    efficientnet_b4,
    "efficientnet_b7":    efficientnet_b7,
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
}

_model_cache: dict = {}


# ── Model helpers ─────────────────────────────────────────────────────────

def discover_models():
    """Return [{name, file}, …] for every .tar found in models/."""
    if not os.path.isdir(MODELS_DIR):
        return []
    result = []
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith(".tar"):
            continue
        arch = fname.split("_weights")[0] if "_weights" in fname else fname[:-4]
        result.append({"name": arch, "file": fname})
    return result


def _replace_classifier(model, n_classes):
    """Replace the final linear layer with one sized for n_classes."""
    # ResNet / GoogLeNet / ShuffleNet — model.fc
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        return
    # DenseNet — model.classifier (plain Linear)
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, torch.nn.Linear):
            model.classifier = torch.nn.Linear(clf.in_features, n_classes)
            return
        # EfficientNet / MobileNet — model.classifier (Sequential)
        if isinstance(clf, torch.nn.Sequential):
            for i in reversed(range(len(clf))):
                if isinstance(clf[i], torch.nn.Linear):
                    clf[i] = torch.nn.Linear(clf[i].in_features, n_classes)
                    return
    raise RuntimeError(
        f"Cannot locate the classifier layer in {type(model).__name__}. "
        "Add a custom replacement in _replace_classifier()."
    )


def _is_timm_state_dict(state: dict) -> bool:
    """True when the first few keys follow timm naming (conv_stem / blocks.X)."""
    sample = list(state.keys())[:10]
    return any(k.startswith(("conv_stem.", "blocks.", "bn1.")) for k in sample)


def _build_and_load(arch_name: str, filepath: str):
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) \
            if isinstance(checkpoint, dict) else checkpoint

    if _is_timm_state_dict(state):
        # Checkpoint was saved with the timm library — use timm to rebuild it.
        try:
            import timm
        except ImportError:
            raise RuntimeError(
                f"'{arch_name}' checkpoint uses timm weight naming. "
                "Install timm with:  pip install timm"
            )
        model = timm.create_model(arch_name, num_classes=N_CLASSES)
        model.load_state_dict(state)
        model.eval()
        return model

    # torchvision path
    if arch_name not in ARCH_REGISTRY:
        raise ValueError(
            f"Architecture '{arch_name}' is not in ARCH_REGISTRY. "
            "Add it to support this model file."
        )
    model = ARCH_REGISTRY[arch_name]()
    _replace_classifier(model, N_CLASSES)
    model.load_state_dict(state)
    model.eval()
    return model


def get_model(arch_name: str):
    """Return a cached model, loading from disk on first call."""
    if arch_name not in _model_cache:
        index = {e["name"]: e for e in discover_models()}
        if arch_name not in index:
            raise ValueError(f"No .tar file found for '{arch_name}' in {MODELS_DIR}.")
        filepath = os.path.join(MODELS_DIR, index[arch_name]["file"])
        print(f"Loading {arch_name} …", flush=True)
        _model_cache[arch_name] = _build_and_load(arch_name, filepath)
        print(f"{arch_name} ready.", flush=True)
    return _model_cache[arch_name]


# ── Class names ───────────────────────────────────────────────────────────

def load_class_names():
    with open(IDX_TO_SID_PATH, encoding="utf-8") as f:
        idx_to_sid = json.load(f)
    with open(SID_TO_NAME_PATH, encoding="utf-8") as f:
        sid_to_name = json.load(f)
    return {
        int(k): sid_to_name.get(str(v), f"Species {k}")
        for k, v in idx_to_sid.items()
    }


CLASS_NAMES = load_class_names()
print(f"Class names loaded — {len(CLASS_NAMES)} species.", flush=True)


def load_common_names():
    """Parse flower-names.txt → {normalized_latin: {en, tr}}.
    File format per line:  INDEX  Genus species [author]  | English  | Turkish
    Key is lowercase 'genus species' (first two words) for reliable matching
    against model output that may include author abbreviations.
    """
    path = os.path.join(BASE_DIR, "flower-names.txt")
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            tokens = parts[0].split()
            if len(tokens) < 2:
                continue
            latin = " ".join(tokens[1:])          # drop leading index number
            en    = parts[1].strip()
            tr    = parts[2].strip()
            words = latin.split()
            key   = (words[0] + " " + words[1]).lower() if len(words) >= 2 else latin.lower()
            result[key] = {
                "en": en  if en  != "—" else "",
                "tr": tr  if tr  != "—" else "",
            }
    return result


COMMON_NAMES = load_common_names()
print(f"Common names loaded — {len(COMMON_NAMES)} entries.", flush=True)

# ── Image transform (TTA: 5 crops × 2 flips = 10 views) ──────────────────

TTA_TRANSFORM = transforms.Compose([
    transforms.Resize(384),
    transforms.FiveCrop(224),
    transforms.Lambda(
        lambda crops: torch.stack([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                transforms.ToTensor()(c)
            )
            for crop in crops
            for c in (crop, transforms.functional.hflip(crop))
        ])
    ),
])


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/names")
def get_names():
    """Return the common-name lookup dict for frontend language switching."""
    return jsonify(COMMON_NAMES)


@app.route("/models")
def list_models():
    """Return the list of available models discovered from models/."""
    return jsonify({"models": discover_models()})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open image"}), 400

    model_name = request.form.get("model_name", "").strip()
    if not model_name:
        models = discover_models()
        if not models:
            return jsonify({"error": "No models found in the models/ directory."}), 500
        model_name = models[0]["name"]

    try:
        model = get_model(model_name)
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    batch = TTA_TRANSFORM(img)
    with torch.no_grad():
        logits = model(batch)
        probs  = F.softmax(logits, dim=1).mean(dim=0)

    top_probs, top_indices = torch.topk(probs, TOP_K)
    results = [
        {
            "rank":       i + 1,
            "species":    CLASS_NAMES.get(idx.item(), f"Species {idx.item()}"),
            "confidence": round(prob.item() * 100, 2),
        }
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]

    top_conf = results[0]["confidence"]
    warning = (
        "Low confidence — the plant may not be in the 1,081 supported species, "
        "or the image quality / angle differs too much from training photos."
        if top_conf < CONFIDENCE_WARN_THRESHOLD
        else None
    )

    return jsonify({"predictions": results, "warning": warning, "model": model_name})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
