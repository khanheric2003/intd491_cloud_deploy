"use client";

import { useState } from "react";

function FairnessMetricCard({
  label,
  value,
  threshold,
  status,
  description
}: {
  label: string;
  value: string;
  threshold: string;
  status: "fair" | "warn" | "unfair";
  description: string;
}) {
  const statusColors = {
    fair: "badge-fair",
    warn: "badge-warn",
    unfair: "badge-unfair"
  };

  const statusTexts = {
    fair: "Fair",
    warn: "Moderate",
    unfair: "Unfair"
  };

  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div style={{ marginTop: "8px" }}>
        <span className={`badge ${statusColors[status]}`}>{statusTexts[status]}</span>
      </div>
      <p style={{ fontSize: "12px", color: "#64748b", margin: "8px 0 0 0" }}>
        Threshold: {threshold}
      </p>
    </div>
  );
}

type FairnessStatus = "fair" | "warn" | "unfair";

interface ModelFairnessData {
  fprGap: { value: string; status: FairnessStatus };
  fnrGap: { value: string; status: FairnessStatus };
  tprGap: { value: string; status: FairnessStatus };
  accuracy: { value: string; status: FairnessStatus };
  errorRates: {
    fpr: { label: string; value: number }[];
    fnr: { label: string; value: number }[];
  };
}

// All numbers directly from your Colab notebook outputs
const modelFairnessData: Record<string, Record<string, ModelFairnessData>> = {
  "Logistic Regression": {
    race: {
      fprGap:    { value: "+0.163", status: "unfair" },
      fnrGap:    { value: "−0.339", status: "unfair" },
      tprGap:    { value: "+0.339", status: "unfair" },
      accuracy:  { value: "68.6%",  status: "warn"   },
      errorRates: {
        fpr: [
          { label: "African-American", value: 38.5 },
          { label: "Caucasian",        value: 22.2 },
          { label: "Hispanic",         value: 26.5 },
          { label: "Other",            value: 17.0 },
        ],
        fnr: [
          { label: "African-American", value: 20.8 },
          { label: "Caucasian",        value: 54.7 },
          { label: "Hispanic",         value: 48.9 },
          { label: "Other",            value: 55.6 },
        ],
      },
    },
    sex: {
      fprGap:    { value: "−0.041", status: "fair" },
      fnrGap:    { value: "+0.062", status: "fair" },
      tprGap:    { value: "−0.062", status: "fair" },
      accuracy:  { value: "68.6%",  status: "warn" },
      errorRates: {
        fpr: [
          { label: "Male",   value: 29.8 },
          { label: "Female", value: 33.9 },
        ],
        fnr: [
          { label: "Male",   value: 34.1 },
          { label: "Female", value: 27.9 },
        ],
      },
    },
  },

  "Random Forest": {
    race: {
      fprGap:    { value: "+0.146", status: "unfair" },
      fnrGap:    { value: "−0.312", status: "unfair" },
      tprGap:    { value: "+0.312", status: "unfair" },
      accuracy:  { value: "68.5%",  status: "warn"   },
      errorRates: {
        fpr: [
          { label: "African-American", value: 37.5 },
          { label: "Caucasian",        value: 22.9 },
          { label: "Hispanic",         value: 22.4 },
          { label: "Other",            value: 14.9 },
        ],
        fnr: [
          { label: "African-American", value: 22.8 },
          { label: "Caucasian",        value: 54.0 },
          { label: "Hispanic",         value: 46.7 },
          { label: "Other",            value: 55.6 },
        ],
      },
    },
    sex: {
      fprGap:    { value: "−0.038", status: "fair" },
      fnrGap:    { value: "+0.055", status: "fair" },
      tprGap:    { value: "−0.055", status: "fair" },
      accuracy:  { value: "68.5%",  status: "warn" },
      errorRates: {
        fpr: [
          { label: "Male",   value: 28.4 },
          { label: "Female", value: 32.2 },
        ],
        fnr: [
          { label: "Male",   value: 33.2 },
          { label: "Female", value: 27.7 },
        ],
      },
    },
  },

  "RF — Threshold Adjusted": {
    race: {
      fprGap:    { value: "−0.002", status: "fair" },
      fnrGap:    { value: "−0.104", status: "warn" },
      tprGap:    { value: "+0.104", status: "warn" },
      accuracy:  { value: "66.2%",  status: "warn" },
      errorRates: {
        fpr: [
          { label: "African-American", value: 22.7 },
          { label: "Caucasian",        value: 22.9 },
          { label: "Hispanic",         value: 24.5 },
          { label: "Other",            value: 21.3 },
        ],
        fnr: [
          { label: "African-American", value: 43.6 },
          { label: "Caucasian",        value: 54.0 },
          { label: "Hispanic",         value: 46.7 },
          { label: "Other",            value: 50.0 },
        ],
      },
    },
    sex: {
      fprGap:    { value: "−0.012", status: "fair" },
      fnrGap:    { value: "+0.031", status: "fair" },
      tprGap:    { value: "−0.031", status: "fair" },
      accuracy:  { value: "66.2%",  status: "warn" },
      errorRates: {
        fpr: [
          { label: "Male",   value: 27.1 },
          { label: "Female", value: 28.3 },
        ],
        fnr: [
          { label: "Male",   value: 36.8 },
          { label: "Female", value: 33.7 },
        ],
      },
    },
  },

  "Fairlearn — EqualizedOdds": {
    race: {
      fprGap:    { value: "+0.107", status: "warn"   },
      fnrGap:    { value: "−0.127", status: "warn"   },
      tprGap:    { value: "+0.127", status: "warn"   },
      accuracy:  { value: "61.6%",  status: "unfair" },
      errorRates: {
        fpr: [
          { label: "African-American", value: 41.1 },
          { label: "Caucasian",        value: 30.5 },
          { label: "Hispanic",         value: 28.6 },
          { label: "Other",            value: 23.4 },
        ],
        fnr: [
          { label: "African-American", value: 38.3 },
          { label: "Caucasian",        value: 50.9 },
          { label: "Hispanic",         value: 44.4 },
          { label: "Other",            value: 61.1 },
        ],
      },
    },
    sex: {
      fprGap:    { value: "−0.021", status: "fair" },
      fnrGap:    { value: "+0.038", status: "fair" },
      tprGap:    { value: "−0.038", status: "fair" },
      accuracy:  { value: "61.6%",  status: "warn" },
      errorRates: {
        fpr: [
          { label: "Male",   value: 31.2 },
          { label: "Female", value: 33.3 },
        ],
        fnr: [
          { label: "Male",   value: 39.4 },
          { label: "Female", value: 35.6 },
        ],
      },
    },
  },
};

type DebiasMethod = "proxyVariableRemoval" | "thresholdAdjustment" | "fairlearnEO";

interface DebiasData {
  label: string;
  type: string;
  accuracy: string;
  accuracyPrevious: string;
  fprGapBefore: string;
  fprGapAfter: string;
  fnrGapBefore: string;
  fnrGapAfter: string;
  aucBefore: string;
  aucAfter: string;
  description: string;
  recommendation: string;
}

// All numbers from your notebook
const debiasMethodData: Record<DebiasMethod, DebiasData> = {
  proxyVariableRemoval: {
    label: "Proxy Variable Removal",
    type: "Pre-processing",
    accuracy: "56.3%",
    accuracyPrevious: "68.5%",
    fprGapBefore: "+0.146",
    fprGapAfter: "+0.103",
    fnrGapBefore: "−0.312",
    fnrGapAfter: "−0.148",
    aucBefore: "0.727",
    aucAfter: "0.598",
    description:
      "Removes priors_count and juvenile history features before training. Reduces the racial FPR gap by 30% but costs 12.2% accuracy — proving these features encode racial bias as proxy variables.",
    recommendation:
      "Use when interpretability and avoiding proxy discrimination is the priority. Accept the accuracy trade-off as the cost of removing biased features."
  },
  thresholdAdjustment: {
    label: "Threshold Adjustment",
    type: "Post-processing",
    accuracy: "66.2%",
    accuracyPrevious: "68.5%",
    fprGapBefore: "+0.146",
    fprGapAfter: "−0.002",
    fnrGapBefore: "−0.312",
    fnrGapAfter: "−0.104",
    aucBefore: "0.727",
    aucAfter: "0.727",
    description:
      "Raises the classification threshold for African-American defendants from 0.5 to 0.613, requiring greater confidence before flagging someone high risk. Nearly eliminates the FPR gap at only 2.3% accuracy cost. Note: uses race at decision time, which is ethically contested.",
    recommendation:
      "Best numerical outcome, but legally and ethically controversial since race is used explicitly at prediction time. Consider Fairlearn for a more defensible deployment."
  },
  fairlearnEO: {
    label: "Fairlearn EqualizedOdds",
    type: "In-processing",
    accuracy: "61.6%",
    accuracyPrevious: "68.5%",
    fprGapBefore: "+0.146",
    fprGapAfter: "+0.107",
    fnrGapBefore: "−0.312",
    fnrGapAfter: "−0.127",
    aucBefore: "0.727",
    aucAfter: "N/A",
    description:
      "Trains a model that explicitly minimizes both FPR and FNR gaps simultaneously using an EqualizedOdds constraint. Unlike threshold adjustment, race is not used at prediction time — fairness is achieved during training itself, making this more legally and ethically defensible.",
    recommendation:
      "Most defensible deployment strategy. Achieves meaningful fairness improvements on both FPR and FNR without using race at decision time, at a moderate 6.9% accuracy cost."
  }
};

export default function FairnessPage() {
  const [protectedAttr, setProtectedAttr] = useState("race");
  const [selectedModel, setSelectedModel] = useState("Random Forest");
  const [debiasMethod, setDebiasMethod] = useState<DebiasMethod>("thresholdAdjustment");

  const currentData = modelFairnessData[selectedModel][protectedAttr];
  const currentDebiasData = debiasMethodData[debiasMethod];

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Fairness Analysis</p>
        <h1>Evaluating Algorithmic Fairness</h1>
        <p>
          Examine fairness metrics and disparities across demographic groups.
          Understand how different models impact fairness in recidivism prediction,
          and explore debiasing strategies. All results are from our trained models
          on the COMPAS Broward County dataset.
        </p>
      </section>

      <div className="section-grid">

        {/* Controls */}
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Fairness Controls</h2>
              <p>Select protected attribute and model</p>
            </div>
            <div className="input-form">
              <div className="form-group">
                <label className="form-label">Protected Attribute</label>
                <select
                  className="form-select"
                  value={protectedAttr}
                  onChange={(e) => setProtectedAttr(e.target.value)}
                >
                  <option value="race">Race</option>
                  <option value="sex">Sex</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Model</label>
                <select
                  className="form-select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="Logistic Regression">Logistic Regression (baseline)</option>
                  <option value="Random Forest">Random Forest (baseline)</option>
                  <option value="RF — Threshold Adjusted">RF — Threshold Adjusted</option>
                  <option value="Fairlearn — EqualizedOdds">Fairlearn — EqualizedOdds</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* Fairness Metric Cards */}
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Fairness Metrics</h2>
              <p>
                Key fairness indicators for {selectedModel} —{" "}
                {protectedAttr === "race" ? "Race (African-American vs Caucasian)" : "Sex (Male vs Female)"}
              </p>
            </div>
            <div className="metrics-grid">
              <FairnessMetricCard
                label="FPR Gap (AA − White)"
                value={currentData.fprGap.value}
                threshold="< ±0.05 is fair"
                status={currentData.fprGap.status}
                description="False positive rate difference between groups"
              />
              <FairnessMetricCard
                label="FNR Gap (AA − White)"
                value={currentData.fnrGap.value}
                threshold="< ±0.05 is fair"
                status={currentData.fnrGap.status}
                description="False negative rate difference between groups"
              />
              <FairnessMetricCard
                label="TPR Gap (AA − White)"
                value={currentData.tprGap.value}
                threshold="< ±0.05 is fair"
                status={currentData.tprGap.status}
                description="True positive rate (recall) difference between groups"
              />
              <FairnessMetricCard
                label="Model Accuracy"
                value={currentData.accuracy.value}
                threshold="~65% baseline"
                status={currentData.accuracy.status}
                description="Overall classification accuracy on test set"
              />
            </div>
          </div>
        </section>

        {/* Error Rate Charts */}
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Error Rates by Demographic Group</h2>
              <p>
                False Positive Rate and False Negative Rate for{" "}
                {selectedModel} — {protectedAttr === "race" ? "by Race" : "by Sex"}
              </p>
            </div>

            <div style={{ marginBottom: "24px" }}>
              <p style={{ fontSize: "13px", fontWeight: 500, color: "#1e293b", marginBottom: "12px" }}>
                False Positive Rate (flagged high risk when they would NOT reoffend)
              </p>
              {currentData.errorRates.fpr.map((item) => (
                <div key={item.label} style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "10px" }}>
                  <span style={{ fontSize: "12px", color: "#64748b", width: "150px", flexShrink: 0 }}>{item.label}</span>
                  <div style={{ flex: 1, height: "20px", background: "#f1f5f9", borderRadius: "4px", overflow: "hidden" }}>
                    <div style={{
                      width: `${(item.value / 60) * 100}%`,
                      height: "100%",
                      background: item.label === "African-American" || item.label === "Male" ? "#ef4444" : "#378ADD",
                      borderRadius: "4px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      paddingRight: "6px"
                    }}>
                      <span style={{ fontSize: "11px", color: "white", fontWeight: 500 }}>{item.value.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div>
              <p style={{ fontSize: "13px", fontWeight: 500, color: "#1e293b", marginBottom: "12px" }}>
                False Negative Rate (missed as low risk when they DID reoffend)
              </p>
              {currentData.errorRates.fnr.map((item) => (
                <div key={item.label} style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "10px" }}>
                  <span style={{ fontSize: "12px", color: "#64748b", width: "150px", flexShrink: 0 }}>{item.label}</span>
                  <div style={{ flex: 1, height: "20px", background: "#f1f5f9", borderRadius: "4px", overflow: "hidden" }}>
                    <div style={{
                      width: `${(item.value / 70) * 100}%`,
                      height: "100%",
                      background: item.label === "Caucasian" || item.label === "Female" ? "#f59e0b" : "#94a3b8",
                      borderRadius: "4px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      paddingRight: "6px"
                    }}>
                      <span style={{ fontSize: "11px", color: "white", fontWeight: 500 }}>{item.value.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {protectedAttr === "race" && (
              <div style={{
                marginTop: "16px",
                padding: "10px 12px",
                background: "#fef3c7",
                borderLeft: "3px solid #f59e0b",
                borderRadius: "0 6px 6px 0",
                fontSize: "12px",
                color: "#92400e"
              }}>
                <strong>Key finding:</strong> African-American defendants who did NOT reoffend are flagged
                high risk at a substantially higher rate than Caucasian defendants. White defendants who
                DID reoffend are missed at more than twice the rate — meaning they receive more benefit of the doubt.
              </div>
            )}
          </div>
        </section>

        {/* Debiasing Comparison */}
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Debiasing Methods Comparison</h2>
              <p>Three mitigation strategies tested — all results from our trained models</p>
            </div>

            <div style={{ marginBottom: "16px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
              {(Object.entries(debiasMethodData) as [DebiasMethod, DebiasData][]).map(([key, method]) => (
                <button
                  key={key}
                  onClick={() => setDebiasMethod(key)}
                  style={{
                    padding: "8px 14px",
                    borderRadius: "6px",
                    border: debiasMethod === key ? "2px solid #0ea5e9" : "1px solid #e2e8f0",
                    background: debiasMethod === key ? "#dbeafe" : "white",
                    color: debiasMethod === key ? "#0c4a6e" : "#374151",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: 500
                  }}
                >
                  {method.label}
                  <span style={{
                    marginLeft: "6px",
                    fontSize: "11px",
                    padding: "2px 6px",
                    borderRadius: "4px",
                    background: debiasMethod === key ? "#bfdbfe" : "#f1f5f9",
                    color: debiasMethod === key ? "#1e40af" : "#64748b"
                  }}>
                    {method.type}
                  </span>
                </button>
              ))}
            </div>

            <div style={{
              padding: "12px 14px",
              background: "#f0f9ff",
              borderRadius: "8px",
              borderLeft: "4px solid #0ea5e9",
              marginBottom: "16px",
              fontSize: "13px",
              color: "#0c4a6e",
              lineHeight: "1.6"
            }}>
              {currentDebiasData.description}
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #e2e8f0" }}>
                    <th style={{ padding: "10px 8px", textAlign: "left", fontSize: "12px", color: "#64748b" }}>Metric</th>
                    <th style={{ padding: "10px 8px", textAlign: "right", fontSize: "12px", color: "#64748b" }}>Baseline RF</th>
                    <th style={{ padding: "10px 8px", textAlign: "right", fontSize: "12px", color: "#64748b" }}>After {currentDebiasData.label}</th>
                    <th style={{ padding: "10px 8px", textAlign: "right", fontSize: "12px", color: "#64748b" }}>Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "10px 8px", fontWeight: 500 }}>Accuracy</td>
                    <td style={{ padding: "10px 8px", textAlign: "right" }}>{currentDebiasData.accuracyPrevious}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right" }}>{currentDebiasData.accuracy}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#ef4444", fontWeight: 500 }}>
                      {debiasMethod === "proxyVariableRemoval" ? "−12.2%" : debiasMethod === "thresholdAdjustment" ? "−2.3%" : "−6.9%"}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "10px 8px", fontWeight: 500 }}>AUC</td>
                    <td style={{ padding: "10px 8px", textAlign: "right" }}>{currentDebiasData.aucBefore}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right" }}>{currentDebiasData.aucAfter}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#64748b" }}>
                      {debiasMethod === "proxyVariableRemoval" ? "−0.129" : debiasMethod === "thresholdAdjustment" ? "0.000" : "N/A"}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "10px 8px", fontWeight: 500 }}>FPR Gap (AA − White)</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#ef4444" }}>{currentDebiasData.fprGapBefore}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#10b981", fontWeight: 500 }}>{currentDebiasData.fprGapAfter}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#10b981", fontWeight: 500 }}>
                      {debiasMethod === "proxyVariableRemoval" ? "−0.043" : debiasMethod === "thresholdAdjustment" ? "−0.148" : "−0.039"}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: "10px 8px", fontWeight: 500 }}>FNR Gap (AA − White)</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#ef4444" }}>{currentDebiasData.fnrGapBefore}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#10b981", fontWeight: 500 }}>{currentDebiasData.fnrGapAfter}</td>
                    <td style={{ padding: "10px 8px", textAlign: "right", color: "#10b981", fontWeight: 500 }}>
                      {debiasMethod === "proxyVariableRemoval" ? "+0.164" : debiasMethod === "thresholdAdjustment" ? "+0.208" : "+0.185"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style={{
              marginTop: "14px",
              padding: "10px 12px",
              background: "#dcfce7",
              borderLeft: "3px solid #10b981",
              borderRadius: "0 6px 6px 0",
              fontSize: "12px",
              color: "#166534"
            }}>
              <strong>Recommendation:</strong> {currentDebiasData.recommendation}
            </div>
          </div>
        </section>

        {/* Key Findings */}
        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Key Findings</h2>
              <p>What our fairness audit shows</p>
            </div>

            <div style={{ display: "grid", gap: "12px", marginTop: "16px" }}>
              <div style={{
                padding: "14px 16px",
                backgroundColor: "#fef3c7",
                borderRadius: "8px",
                borderLeft: "4px solid #f59e0b"
              }}>
                <div style={{ fontWeight: 500, color: "#92400e", marginBottom: "4px" }}>
                  Racial disparity confirmed
                </div>
                <p style={{ margin: 0, fontSize: "13px", color: "#92400e", lineHeight: "1.6" }}>
                  African-American defendants are flagged high risk 37.5% of the time when they would not reoffend,
                  vs 22.9% for Caucasian defendants — a gap of +0.146 that is consistent across both
                  Random Forest and Logistic Regression baselines.
                </p>
              </div>

              <div style={{
                padding: "14px 16px",
                backgroundColor: "#dcfce7",
                borderRadius: "8px",
                borderLeft: "4px solid #10b981"
              }}>
                <div style={{ fontWeight: 500, color: "#166534", marginBottom: "4px" }}>
                  Mitigation is possible but involves trade-offs
                </div>
                <p style={{ margin: 0, fontSize: "13px", color: "#166534", lineHeight: "1.6" }}>
                  Threshold adjustment nearly eliminates the FPR gap (−0.002) at only 2.3% accuracy cost.
                  Removing proxy variables reduces the gap 30% but costs 12.2% accuracy.
                  Fairlearn achieves balanced improvements on both FPR and FNR without using race at prediction time.
                </p>
              </div>

              <div style={{
                padding: "14px 16px",
                backgroundColor: "#dbeafe",
                borderRadius: "8px",
                borderLeft: "4px solid #3b82f6"
              }}>
                <div style={{ fontWeight: 500, color: "#1e40af", marginBottom: "4px" }}>
                  No free lunch — Chouldechova&apos;s impossibility confirmed
                </div>
                <p style={{ margin: 0, fontSize: "13px", color: "#1e40af", lineHeight: "1.6" }}>
                  No single model configuration achieves both high accuracy and full fairness simultaneously.
                  Every mitigation involves a trade-off — confirming that algorithmic bias cannot be fully
                  engineered away when base recidivism rates differ between groups.
                </p>
              </div>
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
