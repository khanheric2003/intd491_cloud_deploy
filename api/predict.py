from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler

# Real logistic regression weights exported from trained sklearn model
LR_COEFFICIENTS = [
    -0.6192135452982421,   # age
    0.13796377016946512,   # sex_male
    0.7883347675655511,    # priors_count
    0.03221666590915705,   # juv_fel_count
    -0.010377429937823184, # juv_misd_count
    0.08073173779230307,   # juv_other_count
    0.14324969223331926,   # charge_degree_felony
    -0.09109538971193913,  # age_cat_25-45
    0.11785333858268647,   # age_cat_Greater than 45
    -0.006213531674871821, # age_cat_Less than 25
]
LR_INTERCEPT = 0.0002275770273049927

SCALER_MEAN = [
    34.44338667206806, 0.8091958679359935, 3.2616973870771724,
    0.05651205185335224, 0.08932550131658902, 0.11079602997771926,
    0.6481669029775167, 0.5815272432651408, 0.2039700222807373,
    0.21450273445412193,
]
SCALER_SCALE = [
    11.67534312741904, 0.39293500130594194, 4.7311749065891116,
    0.39369991459648196, 0.4508689995484128, 0.4647585619260441,
    0.4775422168374714, 0.4933085328732901, 0.40294695965044974,
    0.4104769315879261,
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_recidivism(
    age: int,
    sex: str,
    prior_crimes: int,
    juvenile_felonies: int,
    juvenile_misdemeanors: int,
    juvenile_other: int,
    charge_degree: str,
) -> float:
    sex_male = 1 if sex == "Male" else 0
    charge_felony = 1 if charge_degree == "Felony" else 0
    age_cat_25_45 = 1 if 25 <= age <= 45 else 0
    age_cat_gt45 = 1 if age > 45 else 0
    age_cat_lt25 = 1 if age < 25 else 0

    features = [
        age, sex_male, prior_crimes, juvenile_felonies,
        juvenile_misdemeanors, juvenile_other, charge_felony,
        age_cat_25_45, age_cat_gt45, age_cat_lt25,
    ]

    scaled = [(v - SCALER_MEAN[i]) / SCALER_SCALE[i] for i, v in enumerate(features)]

    logit = LR_INTERCEPT
    for i in range(len(scaled)):
        logit += scaled[i] * LR_COEFFICIENTS[i]

    return sigmoid(logit)


def build_prediction_response(body: bytes) -> tuple[int, dict[str, object]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return 400, {"error": "Request body must be valid JSON."}

    if not isinstance(payload, dict):
        return 400, {"error": "Request body must be a JSON object."}

    required = [
        "age", "sex", "prior_crimes", "juvenile_felonies",
        "juvenile_misdemeanors", "juvenile_other", "charge_degree",
    ]
    for field in required:
        if field not in payload:
            return 400, {"error": f"Missing required field: {field}"}

    age = payload["age"]
    if not isinstance(age, (int, float)) or age < 18 or age > 80:
        return 400, {"error": "Age must be between 18 and 80"}

    if payload["sex"] not in ("Male", "Female"):
        return 400, {"error": "Sex must be Male or Female"}

    prior_crimes = payload["prior_crimes"]
    if not isinstance(prior_crimes, (int, float)) or prior_crimes < 0 or prior_crimes > 40:
        return 400, {"error": "Prior crimes must be between 0 and 40"}

    juvenile_felonies = payload["juvenile_felonies"]
    if not isinstance(juvenile_felonies, (int, float)) or juvenile_felonies < 0 or juvenile_felonies > 10:
        return 400, {"error": "Juvenile felonies must be between 0 and 10"}

    if payload["charge_degree"] not in ("Felony", "Misdemeanor"):
        return 400, {"error": "Charge degree must be Felony or Misdemeanor"}

    risk_score = predict_recidivism(
        age=int(age),
        sex=payload["sex"],
        prior_crimes=int(prior_crimes),
        juvenile_felonies=int(juvenile_felonies),
        juvenile_misdemeanors=int(payload.get("juvenile_misdemeanors", 0)),
        juvenile_other=int(payload.get("juvenile_other", 0)),
        charge_degree=payload["charge_degree"],
    )

    risk_level = "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"

    return 200, {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model": "Logistic Regression",
        "input": payload,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        status_code, payload = build_prediction_response(body)
        self._send_json(status_code, payload)

    def do_GET(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "Use POST /api/predict with COMPAS data payload."})
