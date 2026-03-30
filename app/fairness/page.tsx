"use client";

import { useState } from "react";
import { ClassDistributionChart } from "@/components/ClassDistributionChart";
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

const modelFairnessData: Record<string, ModelFairnessData> = {
  "Logistic Regression": {
    disparateImpact: { value: "0.61", status: "unfair" },
    statisticalParity: { value: "-0.16", status: "unfair" },
    equalOpportunity: { value: "-0.10", status: "warn" },
    equalizedOdds: { value: "-0.08", status: "fair" },
    errorRates: { counts: [42.1, 22.8, 29.5, 27.3], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
  },
  "Random Forest": {
    disparateImpact: { value: "0.58", status: "unfair" },
    statisticalParity: { value: "-0.18", status: "unfair" },
    equalOpportunity: { value: "-0.12", status: "warn" },
    equalizedOdds: { value: "-0.09", status: "warn" },
    errorRates: { counts: [44.8, 23.5, 31.2, 28.6], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
  },
  "Decision Tree": {
    disparateImpact: { value: "0.55", status: "unfair" },
    statisticalParity: { value: "-0.20", status: "unfair" },
    equalOpportunity: { value: "-0.14", status: "unfair" },
    equalizedOdds: { value: "-0.11", status: "unfair" },
    errorRates: { counts: [46.3, 24.1, 33.0, 30.2], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
  },
  "XGBoost + Debiasing": {
    disparateImpact: { value: "0.82", status: "fair" },
    statisticalParity: { value: "-0.07", status: "fair" },
    equalOpportunity: { value: "-0.05", status: "fair" },
    equalizedOdds: { value: "-0.04", status: "fair" },
    errorRates: { counts: [28.4, 24.0, 26.1, 25.5], labels: ["AA: FPR", "Cauc: FPR", "Hisp: FPR", "Other: FPR"] },
  },
};

export default function FairnessPage() {
  const [protectedAttr, setProtectedAttr] = useState("race");
  const [selectedModel, setSelectedModel] = useState("Random Forest");

  const currentData = modelFairnessData[selectedModel];

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
                  <option value="XGBoost + Debiasing">XGBoost + Debiasing</option>
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
          <div className="section-card__body two-column">
            <div>
              <div className="section-card__header">
                <h2>Error Rates by Demographic</h2>
                <p>False Positive and False Negative rates</p>
              </div>
              <p className="section-note">
                African Americans experience significantly higher false positive rates (44.8%) 
                compared to Caucasians (23.5%). This disparity suggests the model is more likely 
                to incorrectly flag African Americans as high-risk.
              </p>
            </div>
            <ClassDistributionChart
              counts={currentData.errorRates.counts}
              labels={currentData.errorRates.labels}
            />
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Debiasing Methods Comparison</h2>
              <p>Impact of different debiasing strategies on fairness metrics</p>
            </div>

            <div style={{ overflowX: "auto" }}>
              <table style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "14px"
              }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #e2e8f0" }}>
                    <th style={{ padding: "12px", textAlign: "left" }}>Method</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Accuracy</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>AUC</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>DI</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>Baseline</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>69.1%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.74</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.58</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-unfair">Unfair</span>
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>Reweighting</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>66.8%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.71</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.72</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-warn">Improved</span>
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>Equalized Odds</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>64.2%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.68</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.85</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-fair">Fair</span>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: "12px" }}>Reject Option</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>65.5%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.70</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.81</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-fair">Fair</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="section-note" style={{ marginTop: "16px" }}>
              The Reject Option method (recommended) achieves fairness while maintaining reasonable accuracy. 
              This approach flags borderline predictions for human review instead of automated decisions.
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
