import numpy as np
import base64
import io
import json


def get_shap_explanation(rf_model, scaler, feature_vector: np.ndarray,
                          feature_names: list) -> dict:
    try:
        import shap  
        X_scaled = scaler.transform(feature_vector.reshape(1, -1))
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            raw = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            raw = shap_vals[1] if shap_vals.shape[0] > 1 else shap_vals[0]
        else:
            raw = shap_vals

        sv = raw[0]

        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_value = float(ev)

        impacts = []
        for i in range(len(feature_names)):
            display_val = float(feature_vector[i])
            if feature_names[i] == "AMH(ng/mL)":
                display_val = round(display_val / 7.14, 3)

            impacts.append({
                "feature":   feature_names[i],
                "shap":      round(float(sv[i]), 4),
                "value":     round(display_val, 3),
                "direction": "increases" if sv[i] > 0 else "decreases"
            })

        impacts.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return {
            "success":    True,
            "base_value": round(base_value, 4),
            "top_features": [
                {
                    "feature":   item["feature"],
                    "shap":      item["shap"],
                    "value":     item["value"],
                    "direction": item["direction"],
                }
                for item in impacts[:10]
            ],
            "all_features": [
                {
                    "feature":   item["feature"],
                    "shap":      item["shap"],
                    "value":     item["value"],
                    "direction": item["direction"],
                }
                for item in impacts
            ],
        }

    except ImportError:
        return _fallback_feature_importance(rf_model, feature_vector, feature_names)

    except Exception as e:
        print(f"[POLYCARE] SHAP ERROR: {e}")
        try:
            return _fallback_feature_importance(rf_model, feature_vector, feature_names)
        except Exception as fallback_err:
            print(f"[POLYCARE] SHAP FALLBACK ERROR: {fallback_err}")
            return {"success": False, "base_value": 0.0, "top_features": [], "all_features": []}



def _fallback_feature_importance(rf_model, feature_vector, feature_names):
    importances = rf_model.feature_importances_
    impacts = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        impacts.append({
            "feature":   name,
            "shap":      round(float(imp), 4),
            "value":     round(float(feature_vector[i]), 3),
            "direction": "increases" if imp > 0 else "neutral"
        })
    impacts.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return {"success": True, "top_features": impacts[:10], "all_features": impacts}


def generate_gradcam(cnn_model, image_array: np.ndarray,
                     last_conv_layer: str = 'top_conv') -> str:
    try:
        import tensorflow as tf
        import cv2

        grad_model = tf.keras.models.Model(
            [cnn_model.inputs],
            [cnn_model.get_layer(last_conv_layer).output, cnn_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i].numpy()

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original = (image_array[0] * 255).astype(np.uint8)
        overlay  = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        img_pil = __import__('PIL').Image.fromarray(overlay)
        buffer  = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        return _placeholder_heatmap(str(e))


def _placeholder_heatmap(error_msg: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (224, 224), color=(20, 30, 50))
        draw = ImageDraw.Draw(img)
        draw.text((30, 100), "Grad-CAM\nUnavailable", fill=(0, 212, 170))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""