"use client";

import PlotlyChart from "@/components/PlotlyChart";

type GroupedErrorRateChartProps = {
  data: Array<{
    race: string;
    fpr: number;
    fnr: number;
  }>;
};

export function GroupedErrorRateChart({ data }: GroupedErrorRateChartProps) {
  const fprValues = data.map(d => d.fpr);
  const fnrValues = data.map(d => d.fnr);
  const races = data.map(d => d.race);

  return (
    <PlotlyChart
      className="chart-frame"
      config={{ displayModeBar: false, responsive: true }}
      data={[
        {
          type: "bar",
          x: races,
          y: fprValues,
          name: "False Positive Rate (FPR)",
          marker: {
            color: "#ef4444",
            line: {
              color: "#991b1b",
              width: 1
            }
          },
          hovertemplate: "%{x}<br>FPR: %{y:.1f}%<extra></extra>"
        },
        {
          type: "bar",
          x: races,
          y: fnrValues,
          name: "False Negative Rate (FNR)",
          marker: {
            color: "#f59e0b",
            line: {
              color: "#b45309",
              width: 1
            }
          },
          hovertemplate: "%{x}<br>FNR: %{y:.1f}%<extra></extra>"
        }
      ]}
      layout={{
        autosize: true,
        height: 400,
        margin: { t: 12, l: 48, r: 18, b: 80 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        barmode: "group",
        xaxis: {
          title: { text: "Demographic Group" },
          tickmode: "array",
          tickvals: races,
          tickangle: 0
        },
        yaxis: {
          title: { text: "Error Rate (%)" },
          gridcolor: "rgba(30, 35, 48, 0.12)"
        },
        legend: {
          x: 0.5,
          y: 1.15,
          xanchor: "center",
          yanchor: "top",
          orientation: "h"
        }
      }}
      style={{ width: "100%", height: "400px" }}
    />
  );
}
