
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import anthropic
import os

app = Flask(__name__)

# --- Mock training data ---
# Features: [wind_speed, temp_celsius, height_metres, crew_size, hazards_logged, hours_since_inspection]
X_train = np.array([
    [5,  20, 10, 4, 1, 2],
    [30, 5,  30, 8, 6, 48],
    [10, 25, 5,  3, 0, 1],
    [40, 2,  40, 10, 9, 72],
    [8,  22, 8,  5, 2, 4],
    [25, 8,  20, 7, 5, 24],
    [3,  28, 3,  2, 0, 0],
    [50, -1, 50, 12, 10, 96],
])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # 0=low risk, 1=high risk

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def generate_ai_summary(features: dict, risk_level: str, probability: float) -> str:
    prompt = f"""You are an HSE (Health, Safety & Environment) advisor.

Site conditions:
- Wind speed: {features['wind_speed']} km/h
- Temperature: {features['temp_celsius']}°C
- Working height: {features['height_metres']} metres
- Crew size: {features['crew_size']}
- Hazards logged today: {features['hazards_logged']}
- Hours since last inspection: {features['hours_since_inspection']}

ML model assessment: {risk_level.upper()} RISK ({probability:.0%} probability)

Write a concise 3-sentence site safety briefing that:
1. States the risk level and the most critical contributing factor
2. Gives one immediate action the site supervisor should take
3. References the relevant ISO 45001 clause

Keep it factual and direct."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    required = ["wind_speed", "temp_celsius", "height_metres", "crew_size",
                "hazards_logged", "hours_since_inspection"]

    if not all(k in data for k in required):
        return jsonify({"error": f"Missing fields. Required: {required}"}), 400

    features = np.array([[data[k] for k in required]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    risk_level = "high" if prediction == 1 else "low"

    summary = generate_ai_summary(data, risk_level, probability)

    return jsonify({
        "risk_level": risk_level,
        "risk_probability": round(float(probability), 3),
        "ai_summary": summary,
        "iso_reference": "ISO 45001:2018",
        "recommendation": "Halt operations pending review" if risk_level == "high" else "Proceed with standard precautions"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier", "features": 6})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
