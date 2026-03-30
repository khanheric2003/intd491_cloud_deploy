"use client";

import { useState } from "react";

export default function PredictionPage() {
  const [formData, setFormData] = useState({
    age: 34,
    sex: "Male",
    priorCrimes: 3,
    juvenileFelonies: 0,
    juvenileMisdemeanors: 0,
    juvenileOther: 0,
    chargeDegree: "Felony",
  });

  const [prediction, setPrediction] = useState<{
    probability: number;
    classification: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: isNaN(Number(value)) ? value : Number(value)
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age: formData.age,
          sex: formData.sex,
          prior_crimes: formData.priorCrimes,
          juvenile_felonies: formData.juvenileFelonies,
          juvenile_misdemeanors: formData.juvenileMisdemeanors,
          juvenile_other: formData.juvenileOther,
          charge_degree: formData.chargeDegree,
        })
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setPrediction({
        probability: data.risk_score,
        classification: data.risk_score >= 0.5 ? "Recidivism Likely" : "Recidivism Unlikely"
      });
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Failed to get prediction");
    } finally {
      setLoading(false);
    }
  };

  const getClassColor = (classification: string) => {
    return classification === "Recidivism Likely" ? "metric-red" : "metric-green";
  };

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Prediction Tool</p>
        <h1>Recidivism Risk Assessment</h1>
        <p>
          Enter demographic and criminal history information to generate a risk prediction
          using a logistic regression model trained on the COMPAS Florida dataset.
          This model is race-blind — race is not used as a predictive feature.
        </p>
      </section>

      <div className="predict-grid">
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Input Information</h2>
              <p>Enter the individual's information</p>
            </div>

            <form className="input-form">
              <div className="form-group">
                <label className="form-label">Age</label>
                <input
                  className="form-input"
                  type="number"
                  name="age"
                  min="18"
                  max="80"
                  value={formData.age}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Sex</label>
                <select
                  className="form-select"
                  name="sex"
                  value={formData.sex}
                  onChange={handleInputChange}
                >
                  <option>Male</option>
                  <option>Female</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Prior Crimes Count</label>
                <input
                  className="form-input"
                  type="number"
                  name="priorCrimes"
                  min="0"
                  max="40"
                  value={formData.priorCrimes}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Juvenile Felonies</label>
                <input
                  className="form-input"
                  type="number"
                  name="juvenileFelonies"
                  min="0"
                  max="10"
                  value={formData.juvenileFelonies}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Juvenile Misdemeanors</label>
                <input
                  className="form-input"
                  type="number"
                  name="juvenileMisdemeanors"
                  min="0"
                  max="10"
                  value={formData.juvenileMisdemeanors}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Juvenile Other Offenses</label>
                <input
                  className="form-input"
                  type="number"
                  name="juvenileOther"
                  min="0"
                  max="10"
                  value={formData.juvenileOther}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Charge Degree</label>
                <select
                  className="form-select"
                  name="chargeDegree"
                  value={formData.chargeDegree}
                  onChange={handleInputChange}
                >
                  <option>Felony</option>
                  <option>Misdemeanor</option>
                </select>
              </div>

              <button
                className="button button--primary"
                type="button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Predicting..." : "Get Prediction"}
              </button>
            </form>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Prediction Result</h2>
              <p>Logistic Regression — trained on COMPAS Florida dataset</p>
            </div>

            {prediction ? (
              <>
                <div className={`metric-card ${getClassColor(prediction.classification)}`}>
                  <div className="metric-label">Classification (threshold = 0.5)</div>
                  <div className="metric-value">{prediction.classification}</div>
                  <div className="metric-subtext">
                    Probability of recidivism: {(prediction.probability * 100).toFixed(1)}%
                  </div>
                </div>

                <div style={{ marginTop: "24px" }}>
                  <div className="section-card__header">
                    <h3>Model Details</h3>
                  </div>
                  <p className="section-note">
                    Accuracy: 68.6% | AUC: 0.731 | F1: 0.657
                  </p>
                  <p className="section-note">
                    This prediction uses a real logistic regression model trained on the COMPAS dataset
                    with L2 regularization and balanced class weights. The model is race-blind — race
                    is excluded from the feature set. Key predictive features include prior crimes count,
                    age, and charge degree.
                  </p>
                  <p className="section-note" style={{ marginTop: "8px" }}>
                    Note: This model reflects patterns in historical criminal justice data, which
                    may contain systemic biases. Predictions should not be used as the sole basis
                    for any decisions.
                  </p>
                </div>
              </>
            ) : (
              <div className="chart-placeholder">
                Enter information and click "Get Prediction" to see the risk assessment.
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
