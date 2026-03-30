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
                    <th style={{ padding: "12px", textAlign: "left" }}>Type</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Accuracy</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>AUC</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>FPR Gap</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>FNR Gap</th>
                    <th style={{ padding: "12px", textAlign: "right" }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>Baseline RF</td>
                    <td style={{ padding: "12px" }}>None</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>68.5%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.727</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#ef4444" }}>+0.146</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#ef4444" }}>-0.312</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-unfair">Unfair</span>
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>No proxy variables</td>
                    <td style={{ padding: "12px" }}>Pre-processing</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>56.3%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.598</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#f59e0b" }}>+0.103</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#f59e0b" }}>-0.148</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-warn">Improved</span>
                    </td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid #e2e8f0" }}>
                    <td style={{ padding: "12px" }}>Threshold adjustment</td>
                    <td style={{ padding: "12px" }}>Post-processing</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>66.2%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.727</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#16a34a" }}>-0.002</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#16a34a" }}>-0.104</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-fair">Fair</span>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: "12px" }}>Fairlearn EqualizedOdds</td>
                    <td style={{ padding: "12px" }}>In-processing</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>61.6%</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>0.N/A</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#f59e0b" }}>+0.107</td>
                    <td style={{ padding: "12px", textAlign: "right", color: "#f59e0b" }}>-0.127</td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <span className="badge badge-warn">Improved</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="section-note" style={{ marginTop: "16px" }}>
              Threshold adjustment achieves near-zero FPR gap at only 2.3% accuracy cost - but uses race at decision time,
              which is ethically contested. Fairlearn optimizes during training, avoiding race-based decisions at prediction time.
            </p>
          </div>
        </section>

        <section className="section-card">
          <div className="section-card__body">
            <div className="section-card__header">
              <h2>Key Findings</h2>
              <p>What our fairness audit shows</p>
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
                  Racial disparity confirmed
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Black defendants are flagged high risk 37.5% of the time when they would not reoffend, vs 22.9% for white
                  defendants - a gap of +0.146 consistent across both RF and LR models.
                </p>
              </div>

              <div style={{
                padding: "16px",
                backgroundColor: "#dcfce7",
                borderRadius: "8px",
                borderLeft: "4px solid #10b981"
              }}>
                <div style={{ fontWeight: 500, color: "#1e293b", marginBottom: "4px" }}>
                  Mitigation is possible but costly
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  Threshold adjustment nearly eliminates the FPR gap (-0.002) at only 2.3% accuracy cost. Removing proxy
                  variables reduces the gap 30% but costs 12% accuracy.
                </p>
              </div>

              <div style={{
                padding: "16px",
                backgroundColor: "#dbeafe",
                borderRadius: "8px",
                borderLeft: "4px solid #0ea5e9"
              }}>
                <div style={{ fontWeight: 500, color: "#1e293b", marginBottom: "4px" }}>
                  No free lunch - Chouldechovas impossibility
                </div>
                <p style={{ margin: 0, fontSize: "14px", color: "#64748b" }}>
                  No single configuration achieves both high accuracy and full fairness simultaneously. Every mitigation
                  involves a trade-off, confirming that algorithmic bias cannot be fully engineered away.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
