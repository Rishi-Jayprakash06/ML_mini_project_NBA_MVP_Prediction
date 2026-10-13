import os
from flask import Flask, render_template, request, jsonify
from model_service import MVPPredictionEngine, FEATURE_METADATA, FEATURES, PRESETS

app = Flask(__name__)
engine = MVPPredictionEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metadata")
def get_metadata():
    return jsonify({
        "features": FEATURES,
        "metadata": FEATURE_METADATA,
        "presets": PRESETS
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        input_features = data.get("features", {})
        model_name = data.get("model", "rf")
        mode = data.get("mode", "full")

        result = engine.predict(input_features, model_name=model_name, mode=mode)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting NBA MVP Prediction GUI at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
