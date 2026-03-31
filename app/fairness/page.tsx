"use client";

import { useState } from "react";
import { ClassDistributionChart } from "@/components/ClassDistributionChart";
import { GroupedErrorRateChart } from "@/components/GroupedErrorRateChart";
import { PredictionChart } from "@/components/PredictionChart";

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
  disparateImpact: { value: string; status: FairnessStatus };
  statisticalParity: { value: string; status: FairnessStatus };
  equalOpportunity: { value: string; status: FairnessStatus };
  equalizedOdds: { value: string; status: FairnessStatus };
  errorRates: { counts: number[]; labels: string[] };
}

const modelFairnessData: Record<string, Record<string, ModelFairnessData>> = {
  "Logistic Regression": {
    race: {
      disparateImpact: { value: "0.61", status: "unfair" },
      statisticalParity: { value: "-0.16", status: "unfair" },
      equalOpportunity: { value: "-0.10", status: "warn" },
      equalizedOdds: { value: "-0.08", status: "fair" },
      errorRates: { counts: [42.1, 22.8, 29.5, 27.3], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
    },
    sex: {
      disparateImpact: { value: "0.72", status: "warn" },
      statisticalParity: { value: "-0.08", status: "fair" },
      equalOpportunity: { value: "-0.06", status: "fair" },
      equalizedOdds: { value: "-0.05", status: "fair" },
      errorRates: { counts: [35.2, 18.5], labels: ["Male: FPR", "Female: FPR"] },
    },
  },
  "Random Forest": {
    race: {
      disparateImpact: { value: "0.58", status: "unfair" },
      statisticalParity: { value: "-0.18", status: "unfair" },
      equalOpportunity: { value: "-0.12", status: "warn" },
      equalizedOdds: { value: "-0.09", status: "warn" },
      errorRates: { counts: [44.8, 23.5, 31.2, 28.6], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
    },
    sex: {
      disparateImpact: { value: "0.75", status: "warn" },
      statisticalParity: { value: "-0.09", status: "fair" },
      equalOpportunity: { value: "-0.07", status: "fair" },
      equalizedOdds: { value: "-0.06", status: "fair" },
      errorRates: { counts: [37.5, 19.8], labels: ["Male: FPR", "Female: FPR"] },
    },
  },
  "Decision Tree": {
    race: {
      disparateImpact: { value: "0.55", status: "unfair" },
      statisticalParity: { value: "-0.20", status: "unfair" },
      equalOpportunity: { value: "-0.14", status: "unfair" },
      equalizedOdds: { value: "-0.11", status: "unfair" },
      errorRates: { counts: [46.3, 24.1, 33.0, 30.2], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
    },
    sex: {
      disparateImpact: { value: "0.68", status: "warn" },
      statisticalParity: { value: "-0.11", status: "warn" },
      equalOpportunity: { value: "-0.09", status: "warn" },
      equalizedOdds: { value: "-0.08", status: "warn" },
      errorRates: { counts: [39.2, 21.3], labels: ["Male: FPR", "Female: FPR"] },
    },
  },
};

type DebiasMethod = "proxyVariableRemoval" | "thresholdAdjustment";

interface DebiasData {
  label: string;
  accuracy: string;
  accuracyPrevious: string;
  fprGapBefore: string;
  fprGapAfter: string;
  aucBefore: string;
  aucAfter: string;
  description: string;
}

const debiasMethodData: Record<DebiasMethod, DebiasData> = {
  proxyVariableRemoval: {
    label: "Proxy Variable Removal",
    accuracy: "56.3%",
    accuracyPrevious: "68.5%",
    fprGapBefore: "+0.146",
    fprGapAfter: "+0.103",
    aucBefore: "0.74",
    aucAfter: "0.62",
    description: "Removes race/sex from model features. Achieves fairness but significant accuracy loss (12.2%)."
  },
  thresholdAdjustment: {
    label: "Threshold Adjustment",
    accuracy: "66.2%",
    accuracyPrevious: "68.5%",
    fprGapBefore: "+0.146",
    fprGapAfter: "-0.002",
    aucBefore: "0.74",
    aucAfter: "0.73",
    description: "Adjusts decision threshold per demographic group. Minimal accuracy loss (2.3%) while achieving fairness."
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
          and explore debiasing strategies.
        </p>
      </section>

      <div className="section-grid">
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
                  <option value="Logistic Regression">Logistic Regression</option>
                  <option value="Random Forest">Random Forest</option>
                  <option value="Decision Tree">Decision Tree</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Fairness Metrics</h2>
              <p>Key fairness indicators for {selectedModel}</p>
            </div>

            <div className="metrics-grid">
              <FairnessMetricCard
                label="Disparate Impact"
                value={currentData.disparateImpact.value}
                threshold="≥0.8"
                status={currentData.disparateImpact.status}
                description="Ratio of negative predictions for protected vs unprotected group"
              />
              <FairnessMetricCard
                label="Statistical Parity Diff"
                value={currentData.statisticalParity.value}
                threshold="±0.1"
                status={currentData.statisticalParity.status}
                description="Difference in positive prediction rates"
              />
              <FairnessMetricCard
                label="Equal Opportunity Diff"
                value={currentData.equalOpportunity.value}
                threshold="±0.1"
                status={currentData.equalOpportunity.status}
                description="Difference in true positive rates"
              />
              <FairnessMetricCard
                label="Equalized Odds Diff"
                value={currentData.equalizedOdds.value}
                threshold="±0.1"
                status={currentData.equalizedOdds.status}
                description="Difference in FPR and FNR"
              />
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Error Rates by Demographic: FPR & FNR</h2>
              <p>False Positive Rate (red) and False Negative Rate (amber) across demographic groups</p>
            </div>
            <GroupedErrorRateChart
              data={[
                { race: "All", fpr: 32.3, fnr: 37.4 },
                { race: "Black", fpr: 44.8, fnr: 28.0 },
                { race: "White", fpr: 23.5, fnr: 47.7 }
              ]}
            />
            <p className="section-note" style={{ marginTop: "16px" }}>
              <strong>Key Finding:</strong> Black individuals show 21.3% higher FPR (44.8% vs 23.5%) but 19.7% lower FNR (28.0% vs 47.7%) compared to White individuals. This reflects a systematic bias toward over-flagging in the protected group and under-flagging in the privileged group.
            </p>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Debiasing Methods Comparison</h2>
              <p>Impact of different debiasing strategies on fairness metrics</p>
            </div>

            <div style={{ marginBottom: "16px", display: "flex", gap: "12px" }}>
              {Object.entries(debiasMethodData).map(([key, method]) => (
                <button
                  key={key}
                  onClick={() => setDebiasMethod(key as DebiasMethod)}
                  style={{
                    padding: "10px 16px",
                    borderRadius: "6px",
                    border: debiasMethod === key ? "2px solid #0ea5e9" : "1px solid #e2e8f0",
                    background: debiasMethod === key ? "#dbeafe" : "white",
                    color: debiasMethod === key ? "#0c4a6e" : "#374151",
                    cursor: "pointer",
                    fontSize: "14px",
                    fontWeight: 500,
                    transition: "all 0.2s"
                  }}
                >
                  {method.label}
                </button>
              ))}
            </div>

            <div style={{
              padding: "14px 16px",
              background: "#f0f9ff",
              borderRadius: "8px",
              borderLeft: "4px solid #0ea5e9",
              marginBottom: "16px",
              fontSize: "13px",
              color: "#0c4a6e"
            }}>
              <strong>Selected Method:</strong> {currentDebiasData.label}<br/>
              {currentDebiasData.description}
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "14px"
              }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #e2e8f0" }}>
                    <th style={{ padding: "12px", textAlign: "left" }}>Metric</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Before Debiasing</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>After {currentDebiasData.label}</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px", fontWeight: 500 }}>Accuracy</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>{currentDebiasData.accuracyPrevious}</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>{currentDebiasData.accuracy}</td>
                    <td style={{ padding: "12px", textAlign: "right", color: debiasMethod === "proxyVariableRemoval" ? "#ef4444" : "#f59e0b", fontWeight: 500 }}>
                      {debiasMethod === "proxyVariableRemoval" ? "-12.2%" : "-2.3%"}
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px", fontWeight: 500 }}>AUC</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>{currentDebiasData.aucBefore}</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>{currentDebiasData.aucAfter}</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#64748b" }}>
                      {debiasMethod === "proxyVariableRemoval" ? "-0.12" : "-0.01"}
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: "12px", fontWeight: 500 }}>FPR Gap (Black-White)</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#ef4444" }}>{currentDebiasData.fprGapBefore}</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#10b981" }}>{currentDebiasData.fprGapAfter}</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#10b981", fontWeight: 500 }}>
                      {debiasMethod === "proxyVariableRemoval" ? "-0.043" : "-0.148"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="section-note" style={{ marginTop: "16px" }}>
              <strong>Recommendation:</strong> {debiasMethod === "thresholdAdjustment" 
                ? "Threshold Adjustment is recommended. It achieves fairness (FPR gap: -0.002) with minimal accuracy loss (2.3%), making it practical for deployment." 
                : "Proxy Variable Removal achieves fairness but at significant accuracy cost (12.2%), making it less suitable for production systems where predictive performance matters."}
            </p>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Key Findings</h2>
            </div>

            <div style={{
              display: "grid",
              gap: "16px",
              marginTop: "16px"
            }}>
              <div style={{
                padding: "16px",
                backgroundColor: "#fef3c7",
                borderRadius: "8px",
                borderLeft: "4px solid #f59e0b"
              }}>
                <div style={{ fontWeight: 500, color: "#1e293b", marginBottom: "4px" }}>
                  Performance Gap
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  The baseline model shows 7.8% disparate impact below the 0.8 fairness threshold, 
                  indicating substantial bias against African Americans.
                </p>
              </div>

              <div style={{
                padding: "16px",
                backgroundColor: "#dcfce7",
                borderRadius: "8px",
                borderLeft: "4px solid #10b981"
              }}>
                <div style={{ fontWeight: 500, color: "#1e293b", marginBottom: "4px" }}>
                  Debiasing Effectiveness
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Equalized Odds post-processing achieves DI of 0.85, meeting fairness criteria, 
                  with only 4.9% accuracy reduction.
                </p>
              </div>

              <div style={{
                padding: "16px",
                backgroundColor: "#dbeafe",
                borderRadius: "8px",
                borderLeft: "4px solid #0ea5e9"
              }}>
                <div style={{ fontWeight: 500, color: "#1e293b", marginBottom: "4px" }}>
                  Recommendation
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Deploy with reject option for borderline cases and implement continuous fairness monitoring.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
