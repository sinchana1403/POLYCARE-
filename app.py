import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"
os.environ["CUDA_VISIBLE_DEVICES"]    = ""    
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import warnings
warnings.filterwarnings("ignore")

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*absl.*")

import sys
import re
import time
import logging
import traceback
import json
from functools import wraps

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'frontend')

sys.path.insert(0, os.path.join(BASE_DIR, 'utils'))

from preprocess import (
    build_feature_vector, build_display_vector,
    compute_risk_level, get_abnormal_features,
    preprocess_image, compute_clinical_override,
    apply_clinical_blend,
    validate_medical_input_ranges,  
    FEATURE_NAMES      
)
from explainability import get_shap_explanation, generate_gradcam



logging.getLogger("werkzeug").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "polycare.log")),
    ],
)
log = logging.getLogger("polycare")


app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   


models = {
    "rf":       None,
    "xgb":      None,
    "scaler":   None,
    "imputer":  None,
    "cnn":      None,
    "metadata": {},
}


def _build_efficientnet_architecture():
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model

    base = EfficientNetB0(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out)


def load_models():
    """Load all trained models from disk with clean, professional logging."""
    log.info("=" * 60)
    log.info("POLYCARE – Model Loading")
    log.info("=" * 60)

    for name, fname in [
        ("rf",      "structured_model.pkl"),
        ("xgb",     "xgb_model.pkl"),
        ("scaler",  "scaler.pkl"),
        ("imputer", "imputer.pkl"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            log.info(f"  [OK] {fname}")
        else:
            log.warning(f"  [MISSING] {fname} — using demo mode fallback")

    cnn_path = os.path.join(MODEL_DIR, "imagef_model.h5")
    if not os.path.exists(cnn_path):
        log.warning("  [MISSING] imagef_model.h5 — CNN disabled")
    else:
        try:
            import absl.logging as absl_log
            absl_log.set_verbosity(absl_log.ERROR)
        except ImportError:
            pass

        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        try:
            models["cnn"] = tf.keras.models.load_model(
                cnn_path, compile=False, custom_objects=None
            )
            log.info("  [OK] imagef_model.h5 (full model load)")
        except Exception as e1:
            log.info(
    "  [INFO] CNN full model load not compatible, using fallback (weights-only)..."
)
            
            try:
                cnn_model = _build_efficientnet_architecture()
                cnn_model.load_weights(cnn_path)
                models["cnn"] = cnn_model
                log.info("  [OK] CNN LOADED (weights-only fallback load)")
            except Exception as e2:
                log.error(
                    f"  [FAILED] CNN could not be loaded — both methods failed.\n"
                    f"    Method 1: {e1}\n"
                    f"    Method 2: {e2}\n"
                    "    Image prediction will be disabled."
                )

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            models["metadata"] = json.load(f)

    log.info("-" * 60)
    status_lines = [
        f"  Random Forest : {'loaded' if models['rf']     else 'MISSING (demo mode)'}",
        f"  XGBoost       : {'loaded' if models['xgb']    else 'MISSING (demo mode)'}",
        f"  Scaler        : {'loaded' if models['scaler'] else 'MISSING'}",
        f"  CNN           : {'loaded' if models['cnn']    else 'MISSING (image disabled)'}",
    ]
    for line in status_lines:
        log.info(line)

    missing_critical = [n for n in ["rf", "scaler"] if models[n] is None]
    if missing_critical:
        log.error(
            f"CRITICAL: {missing_critical} not loaded. "
            "SHAP explainability will be unavailable."
        )
    else:
        log.info("  SHAP          : ready")
    log.info("=" * 60)


def _run_structured_inference(data: dict) -> tuple:
    """
    Build feature vector → scale → run RF + XGBoost → clinical override.
    Returns (results_dict, raw_feat_vec, scaled_feat_vec).
    """
    feat_vec = build_feature_vector(data)

    X_scaled = (
        models["scaler"].transform(feat_vec.reshape(1, -1))
        if models["scaler"] else feat_vec.reshape(1, -1)
    )

    results = {}

    if models["rf"]:
        rf_prob = float(models["rf"].predict_proba(X_scaled)[0][1])
        results["rf"] = {
            "probability": rf_prob,
            "prediction":  int(rf_prob >= 0.5),
        }
    else:
        demo_prob = _demo_heuristic(feat_vec)
        results["rf"] = {"probability": demo_prob, "prediction": int(demo_prob >= 0.5)}

    if models["xgb"]:
        xgb_prob = float(models["xgb"].predict_proba(X_scaled)[0][1])
        results["xgb"] = {
            "probability": xgb_prob,
            "prediction":  int(xgb_prob >= 0.5),
        }
    else:
        results["xgb"] = dict(results["rf"])   

    override = compute_clinical_override(data)
    results["clinical_override"] = override

    return results, feat_vec, X_scaled


def _demo_heuristic(feat_vec: np.ndarray) -> float:
    """Simple rule-based score used when real models are not loaded."""
    feat = dict(zip(FEATURE_NAMES, feat_vec))
    score = 0.0

    bmi = feat.get("BMI", 22)
    if bmi > 30:   score += 0.20
    elif bmi > 25: score += 0.10

    lh  = feat.get("LH(mIU/mL)", 8)
    fsh = feat.get("FSH(mIU/mL)", 6)
    if fsh > 0 and (lh / fsh) > 2:
        score += 0.15

    amh = feat.get("AMH(ng/mL)", 14.28)   
    if amh > 25:  score += 0.15
    elif amh < 7: score -= 0.05

    fl = feat.get("Follicle No. (L)", 6) + feat.get("Follicle No. (R)", 6)
    if fl > 24: score += 0.20
    elif fl > 14: score += 0.10

    for sym in ["Weight gain(Y/N)", "hair growth(Y/N)",
                "Skin darkening (Y/N)", "Pimples(Y/N)"]:
        score += feat.get(sym, 0) * 0.05

    if feat.get("Cycle(R/I)", 4) == 2:
        score += 0.10

    return float(np.clip(score, 0.05, 0.95))


def _compute_shap(feat_vec: np.ndarray) -> dict:
    """Compute SHAP values; return safe empty dict on failure."""
    empty = {"success": False, "top_features": [], "all_features": [], "base_value": 0.0}

    if not (models["rf"] and models["scaler"]):
        missing = [n for n in ["rf", "scaler"] if not models[n]]
        log.warning(f"SHAP skipped — missing: {missing}")
        return empty

    try:
        display_vec = build_display_vector(feat_vec)
        shap_data   = get_shap_explanation(
            models["rf"], models["scaler"], display_vec, FEATURE_NAMES
        )
        shap_data.setdefault("top_features", [])
        shap_data.setdefault("all_features", [])
        shap_data.setdefault("base_value",   0.0)
        if not isinstance(shap_data["top_features"], list):
            shap_data["top_features"] = []
        if not isinstance(shap_data["all_features"], list):
            shap_data["all_features"] = []
        log.info(f"SHAP OK — {len(shap_data['top_features'])} features returned")
        return shap_data
    except Exception as e:
        log.error(f"SHAP computation failed: {e}", exc_info=True)
        return empty


_PDF_PATTERNS = {
    "LH":  [r"(?:LH|luteinizing\s+hormone)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "FSH": [r"(?:FSH|follicle[- ]stimulating\s+hormone)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "AMH": [
        r"(?:AMH|anti[- ]?m[uü]llerian\s+hormone)[^\d]{0,30}([\d]+(?:\.\d+)?)",
        r"(?:MIS)[^\d]{0,20}([\d]+(?:\.\d+)?)",
    ],
    "TSH": [r"(?:TSH|thyroid[- ]stimulating\s+hormone)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "PRL": [r"(?:PRL|prolactin)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "RBS": [
        r"(?:RBS|random\s+blood\s+sugar|glucose\s+random|blood\s+glucose)[^\d]{0,30}([\d]+(?:\.\d+)?)",
        r"(?:glucose)[^\d]{0,20}([\d]+(?:\.\d+)?)",
    ],
    "Hb":  [r"(?:Hb|haemoglobin|hemoglobin)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "BMI": [r"(?:BMI|body\s+mass\s+index)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "bp_systolic": [
        r"(?:systolic|SBP)[^\d]{0,20}([\d]{2,3})",
        r"BP[^\d]{0,10}([\d]{2,3})\s*/\s*[\d]+",
        r"blood\s+pressure[^\d]{0,20}([\d]{2,3})\s*/",
    ],
    "bp_diastolic": [
        r"(?:diastolic|DBP)[^\d]{0,20}([\d]{2,3})",
        r"BP[^\d]{0,10}[\d]{2,3}\s*/\s*([\d]{2,3})",
        r"blood\s+pressure[^\d]{0,20}[\d]{2,3}\s*/\s*([\d]{2,3})",
    ],
    "vitD3": [r"(?:vitamin\s*d3?|25[-\s]?oh\s*d3?|calcidiol)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
    "PRG":  [r"(?:progesterone|PRG|P4)[^\d]{0,30}([\d]+(?:\.\d+)?)"],
}

_PLAUSIBILITY = {
    "LH": (0.1, 200), "FSH": (0.1, 200), "AMH": (0.01, 50),
    "TSH": (0.01, 100), "PRL": (0.1, 500), "RBS": (50, 600),
    "Hb": (3, 25), "BMI": (10, 65), "bp_systolic": (60, 220),
    "bp_diastolic": (40, 150), "vitD3": (1, 200), "PRG": (0.01, 100),
}


def _extract_from_text(text: str) -> dict:
    text_lower = text.lower()
    extracted  = {}
    for field, patterns in _PDF_PATTERNS.items():
        lo, hi = _PLAUSIBILITY.get(field, (0, 1e9))
        for pat in patterns:
            m = re.search(pat, text_lower, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if lo <= val <= hi:
                        extracted[field] = round(val, 3)
                        break
                except (ValueError, IndexError):
                    continue
    return extracted


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Try pypdf → PyPDF2 → pdfplumber → pdfminer in order."""
    import io

    for lib_name, extractor in [
        ("pypdf",      lambda b: _try_pypdf(b)),
        ("PyPDF2",     lambda b: _try_pypdf2(b)),
        ("pdfplumber", lambda b: _try_pdfplumber(b)),
        ("pdfminer",   lambda b: _try_pdfminer(b)),
    ]:
        try:
            text = extractor(pdf_bytes)
            if text and text.strip():
                log.info(f"PDF extracted via {lib_name} ({len(text)} chars)")
                return text
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"{lib_name} failed: {e}")

    log.warning("All PDF extraction methods failed")
    return ""


def _try_pypdf(b):
    import io, pypdf
    r = pypdf.PdfReader(io.BytesIO(b))
    return "\n".join(p.extract_text() or "" for p in r.pages)

def _try_pypdf2(b):
    import io, PyPDF2
    r = PyPDF2.PdfReader(io.BytesIO(b))
    return "\n".join(p.extract_text() or "" for p in r.pages)

def _try_pdfplumber(b):
    import io, pdfplumber
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def _try_pdfminer(b):
    import io
    from pdfminer.high_level import extract_text as pdfminer_extract
    return pdfminer_extract(io.BytesIO(b))


def _validate_ultrasound_image(img_bytes: bytes) -> dict:
    from PIL import Image
    import io

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img.resize((64, 64)), dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        avg_diff   = (np.mean(np.abs(r - g)) +
                      np.mean(np.abs(r - b)) +
                      np.mean(np.abs(g - b))) / 3
        brightness = np.mean(arr)
        log.debug(f"[US-Check] avg_diff={avg_diff:.2f} brightness={brightness:.2f}")

        if avg_diff > 15:
            return {"valid": False,
                    "reason": "Colour image detected — please upload a greyscale ultrasound"}
        if brightness < 10:
            return {"valid": False, "reason": "Image too dark or blank"}
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "reason": f"Image validation error: {e}"}


@app.route("/")
def serve_home():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    fp = os.path.join(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, path if os.path.exists(fp) else "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":  "online",
        "service": "POLYCARE",
        "version": "3.1.0",
        "models_loaded": {
            "random_forest": models["rf"]  is not None,
            "xgboost":       models["xgb"] is not None,
            "cnn":           models["cnn"] is not None,
        },
        "demo_mode": models["rf"] is None,
        "metadata":  models["metadata"],
    })


@app.route("/api/extract_pdf", methods=["POST"])
def extract_pdf():
    try:
        if "pdf" not in request.files:
            return jsonify({"success": False, "error": "No PDF file provided"}), 400

        pdf_file = request.files["pdf"]
        if not pdf_file.filename.lower().endswith(".pdf"):
            return jsonify({"success": False, "error": "File must be a PDF"}), 400

        pdf_bytes = pdf_file.read()
        raw_text  = _pdf_to_text(pdf_bytes)

        if not raw_text.strip():
            return jsonify({
                "success": False,
                "error":   "Could not extract text. Upload a digitally-generated PDF, "
                           "not a scanned image.",
            }), 200

        medical_keywords = ["lab", "report", "blood", "hormone", "amh", "fsh", "lh",
                            "thyroid", "prolactin", "glucose", "haemoglobin"]
        if not any(k in raw_text.lower() for k in medical_keywords):
            return jsonify({
                "success": False,
                "error":   "Document does not appear to be a medical lab report",
            }), 422

        extracted = _extract_from_text(raw_text)
        if len(extracted) < 2:
            return jsonify({
                "success": False,
                "error":   "Fewer than 2 clinical values found. "
                           "Please enter values manually.",
            }), 422

        return jsonify({"success": True, "extracted": extracted, "count": len(extracted)})

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/predict_structured", methods=["POST"])
def predict_structured():
    t0 = time.time()
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        val_result = validate_medical_input_ranges(data)
        if not val_result["valid"]:
            log.warning(f"Validation failed: {val_result['errors']}")
            return jsonify({
                "error":   "Input validation failed",
                "details": val_result["errors"],
            }), 422

        log.info(f"Structured prediction: age={data.get('age','?')} "
                 f"cycle={data.get('cycle','?')} LH={data.get('LH','?')}")

        results, feat_vec, X_scaled = _run_structured_inference(data)

        W_RF, W_XGB = 0.60, 0.40
        model_prob = (W_RF  * results["rf"]["probability"] +
                      W_XGB * results["xgb"]["probability"])

        override  = results["clinical_override"]
        final_prob, override_applied = apply_clinical_blend(model_prob, override)

        if override_applied:
            log.info(f"Clinical blend applied: model={model_prob:.3f} "
                     f"→ blended={final_prob:.3f} "
                     f"(flags={override['count']}, boost={override['boost']})")

        final_pred = int(final_prob >= 0.5)
        risk       = compute_risk_level(final_prob)
        abnormals  = get_abnormal_features(feat_vec)
        shap_data  = _compute_shap(feat_vec)

        elapsed = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "success":         True,
            "prediction":      final_pred,
            "label":           "PCOS Positive" if final_pred else "PCOS Negative",
            "probability":     round(final_prob, 4),
            "confidence_pct":  round(final_prob * 100, 1),
            "risk":            risk,
            "models": {
                "random_forest": results["rf"],
                "xgboost":       results["xgb"],
                "fused":         {"probability": round(final_prob, 4),
                                  "prediction":  final_pred},
            },
            "abnormal_features": abnormals,
            "shap":              shap_data,
            "clinical_override": {
                "count":   override.get("count",  0),
                "flags":   override.get("flags",  []),
                "label":   override.get("label",  ""),
                "applied": override_applied,
            },
            "inference_ms": elapsed,
            "demo_mode":    models["rf"] is None,
        })

    except Exception as e:
        log.error(f"Structured prediction error:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/predict_image", methods=["POST"])
def predict_image():
    t0 = time.time()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file      = request.files["image"]
        img_bytes = file.read()

        if not img_bytes:
            return jsonify({"error": "Empty image file"}), 400

        from PIL import Image
        import io as _io
        try:
            img = Image.open(_io.BytesIO(img_bytes))
            img.verify()                               
            img = Image.open(_io.BytesIO(img_bytes))  
        except Exception:
            return jsonify({"error": "Invalid or corrupted image file"}), 400

        if img.format not in ("JPEG", "PNG"):
            return jsonify({"error": "Only JPEG or PNG images are accepted"}), 400

        if len(img_bytes) < 2_000:
            return jsonify({"error": "Image file is too small (< 2 KB)"}), 400

        us_check = _validate_ultrasound_image(img_bytes)
        if not us_check["valid"]:
            return jsonify({
                "error":  "Invalid image type",
                "detail": us_check["reason"],
                "hint":   "Only greyscale pelvic ultrasound images are accepted.",
            }), 422

        img_array = preprocess_image(img_bytes)

        if models["cnn"]:
            raw_prob = float(models["cnn"].predict(img_array, verbose=0)[0][0])
            img_pred = int(raw_prob >= 0.5)
            gradcam  = generate_gradcam(models["cnn"], img_array)
        else:
            raw_prob = 0.45
            img_pred = 0
            gradcam  = ""
            log.info("CNN not loaded — returning demo image prediction")

        
        confidence_warning = None
        if 0.4 < raw_prob < 0.6:
            confidence_warning = (
                "Low confidence — prediction near decision boundary (0.5). "
                "Results should be interpreted cautiously."
            )

        risk    = compute_risk_level(raw_prob)
        elapsed = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "success":          True,
            "prediction":       img_pred,
            "label":            "PCOS Positive" if img_pred else "PCOS Negative",
            "probability":      round(raw_prob, 4),
            "confidence_pct":   round(raw_prob * 100, 1),
            "risk":             risk,
            "confidence_warning": confidence_warning,
            "gradcam_b64":      gradcam,
            "inference_ms":     elapsed,
            "demo_mode":        models["cnn"] is None,
        })

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/predict_final", methods=["POST"])
def predict_final():
    t0 = time.time()
    try:
        # ── Parse structured data ────────────────────────────────────
        structured_json = request.form.get("structured_data")
        if not structured_json:
            return jsonify({"error": "Missing structured_data field"}), 400
        data = json.loads(structured_json)

        # ── Validate before anything else ────────────────────────────
        val_result = validate_medical_input_ranges(data)
        if not val_result["valid"]:
            log.warning(f"Validation failed (predict_final): {val_result['errors']}")
            return jsonify({
                "error":   "Input validation failed",
                "details": val_result["errors"],
            }), 422

        struct_results, feat_vec, X_scaled = _run_structured_inference(data)

        W_RF, W_XGB = 0.60, 0.40
        struct_prob = (W_RF  * struct_results["rf"]["probability"] +
                       W_XGB * struct_results["xgb"]["probability"])

        img_prob  = None
        gradcam   = ""
        has_image = False

        if "image" in request.files:
            file      = request.files["image"]
            img_bytes = file.read()

            if img_bytes:
                from PIL import Image
                import io as _io
                try:
                    img = Image.open(_io.BytesIO(img_bytes))
                    img.verify()
                    img = Image.open(_io.BytesIO(img_bytes))
                except Exception:
                    return jsonify({"error": "Invalid image file"}), 400

                if img.format not in ("JPEG", "PNG"):
                    return jsonify({"error": "Only JPEG or PNG images accepted"}), 400

                if len(img_bytes) < 2_000:
                    return jsonify({"error": "Image file too small (< 2 KB)"}), 400

                us_check = _validate_ultrasound_image(img_bytes)
                if not us_check["valid"]:
                    return jsonify({
                        "error":  "Invalid image type",
                        "detail": us_check["reason"],
                        "hint":   "Only greyscale pelvic ultrasound images accepted.",
                    }), 422

                img_array = preprocess_image(img_bytes)
                has_image = True

                if models["cnn"]:
                    img_prob = float(
                        models["cnn"].predict(img_array, verbose=0)[0][0]
                    )
                    gradcam  = generate_gradcam(models["cnn"], img_array)
                    log.info(f"CNN prediction: {img_prob:.4f}")
                else:
                    log.info("CNN not loaded — image component skipped in fusion")

        override = struct_results.get("clinical_override", {})
        boost    = override.get("boost")

        if has_image and img_prob is not None:
            W_STRUCT = 0.70 if boost else 0.55
            W_IMAGE  = 0.30 if boost else 0.45
            model_final = W_STRUCT * struct_prob + W_IMAGE * img_prob
            log.info(
                f"Fusion: struct={struct_prob:.3f}×{W_STRUCT} + "
                f"img={img_prob:.3f}×{W_IMAGE} = {model_final:.3f}"
            )
        else:
            model_final = struct_prob
            log.info(f"Structured-only prediction: {model_final:.3f}")

        final_prob, override_applied = apply_clinical_blend(model_final, override)

        final_pred = int(final_prob >= 0.5)
        risk       = compute_risk_level(final_prob)
        abnormals  = get_abnormal_features(feat_vec)
        shap_data  = _compute_shap(feat_vec)

        confidence_warning = None
        if img_prob is not None and 0.4 < img_prob < 0.6:
            confidence_warning = "CNN prediction near decision boundary — interpret cautiously."

        elapsed = round((time.time() - t0) * 1000, 1)

        return jsonify({
            "success":         True,
            "prediction":      final_pred,
            "label":           "PCOS Positive" if final_pred else "PCOS Negative",
            "probability":     round(final_prob, 4),
            "confidence_pct":  round(final_prob * 100, 1),
            "risk":            risk,
            "confidence_warning": confidence_warning,
            "models": {
                "random_forest": struct_results["rf"],
                "xgboost":       struct_results["xgb"],
                "fused":         {"probability": round(final_prob, 4),
                                  "prediction":  final_pred},
            },
            "abnormal_features": abnormals,
            "shap":              shap_data,
            "fusion": {
                "structured_probability": round(struct_prob,   4),
                "image_probability":      round(img_prob, 4) if img_prob is not None else None,
                "final_probability":      round(final_prob, 4),
                "has_image":              has_image,
            },
            "clinical_override": {
                "count":   override.get("count",  0),
                "flags":   override.get("flags",  []),
                "label":   override.get("label",  ""),
                "applied": override_applied,
            },
            "gradcam_b64":  gradcam,
            "inference_ms": elapsed,
        })

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify({"features": FEATURE_NAMES})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large — maximum upload size is 20 MB"}), 413

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    load_models()
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"POLYCARE v3.1 starting on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)