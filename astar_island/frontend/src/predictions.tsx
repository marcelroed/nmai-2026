import { useState } from "react";
import type { InitialState, RoundData, SeedAnalysis } from "./types";
import { MetricCard, MiniMap } from "./viewer";

// ── Prediction class colors (RGB) ────────────────────────────────────────────

const CLASS_COLORS: [number, number, number][] = [
  [194, 199, 208], // 0: Empty
  [255, 156, 67], // 1: Settlement
  [42, 147, 167], // 2: Port
  [145, 43, 47], // 3: Ruin
  [74, 123, 55], // 4: Forest
  [123, 125, 135], // 5: Mountain
];

const CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"];

// ── Color utilities ──────────────────────────────────────────────────────────

function blendDistribution(dist: number[]): string {
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < 6; i++) {
    r += dist[i] * CLASS_COLORS[i][0];
    g += dist[i] * CLASS_COLORS[i][1];
    b += dist[i] * CLASS_COLORS[i][2];
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function argmaxInfo(dist: number[]): { color: string; confidence: number } {
  let maxIdx = 0;
  let maxVal = dist[0];
  for (let i = 1; i < 6; i++) {
    if (dist[i] > maxVal) {
      maxVal = dist[i];
      maxIdx = i;
    }
  }
  const [r, g, b] = CLASS_COLORS[maxIdx];
  return { color: `rgb(${r},${g},${b})`, confidence: maxVal };
}

// ── Scoring (matches Rust scoring.rs) ────────────────────────────────────────

function cellEntropy(dist: number[]): number {
  let h = 0;
  for (const p of dist) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

function cellKL(gt: number[], pred: number[]): number {
  let kl = 0;
  for (let i = 0; i < 6; i++) {
    if (gt[i] > 0) {
      kl += gt[i] * Math.log(gt[i] / Math.max(pred[i], 1e-10));
    }
  }
  return kl;
}

function scorePrediction(prediction: number[][][], groundTruth: number[][][]): number {
  let totalWeightedKL = 0;
  let totalWeight = 0;
  for (let y = 0; y < groundTruth.length; y++) {
    for (let x = 0; x < groundTruth[y].length; x++) {
      const gtCell = groundTruth[y][x];
      const predCell = prediction[y]?.[x];
      if (!predCell) continue;
      const entropy = cellEntropy(gtCell);
      if (entropy < 1e-6) continue;
      const kl = cellKL(gtCell, predCell);
      totalWeightedKL += entropy * kl;
      totalWeight += entropy;
    }
  }
  return totalWeight === 0 ? 0 : totalWeightedKL / totalWeight;
}

function competitionScore(kl: number): number {
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * kl)));
}

function divergenceColor(kl: number): string {
  const t = Math.min(kl / 1.5, 1.0);
  const r = Math.round(t * 214 + (1 - t) * 63);
  const g = Math.round(t * 106 + (1 - t) * 185);
  const b = Math.round(t * 87 + (1 - t) * 80);
  return `rgb(${r},${g},${b})`;
}

function formatDistTooltip(dist: number[], x: number, y: number): string {
  return `(${x}, ${y})\n${dist.map((p, i) => `${CLASS_NAMES[i]}: ${(p * 100).toFixed(1)}%`).join("\n")}`;
}

// ── Map components ───────────────────────────────────────────────────────────

function DistributionMap(props: {
  grid: number[][][];
  mode: "blend" | "argmax";
  label: string;
}) {
  const height = props.grid.length;
  const width = props.grid[0]?.length ?? 0;

  return (
    <div>
      <div className="prediction-map-label">{props.label}</div>
      <svg className="distribution-map" viewBox={`0 0 ${width} ${height}`}>
        {props.grid.flatMap((row, y) =>
          row.map((dist, x) => {
            if (props.mode === "blend") {
              return (
                <rect key={`${y}-${x}`} x={x} y={y} width={1} height={1} fill={blendDistribution(dist)}>
                  <title>{formatDistTooltip(dist, x, y)}</title>
                </rect>
              );
            }
            const { color, confidence } = argmaxInfo(dist);
            return (
              <rect
                key={`${y}-${x}`}
                x={x}
                y={y}
                width={1}
                height={1}
                fill={color}
                opacity={0.3 + 0.7 * confidence}
              >
                <title>{formatDistTooltip(dist, x, y)}</title>
              </rect>
            );
          }),
        )}
      </svg>
    </div>
  );
}

function DivergenceMap(props: { prediction: number[][][]; groundTruth: number[][][] }) {
  const height = props.groundTruth.length;
  const width = props.groundTruth[0]?.length ?? 0;

  return (
    <div>
      <div className="prediction-map-label">Divergence</div>
      <svg className="distribution-map" viewBox={`0 0 ${width} ${height}`}>
        {props.groundTruth.flatMap((row, y) =>
          row.map((gtDist, x) => {
            const predDist = props.prediction[y]?.[x];
            if (!predDist) return null;
            const entropy = cellEntropy(gtDist);
            if (entropy < 1e-6) {
              return (
                <rect key={`${y}-${x}`} x={x} y={y} width={1} height={1} fill="#1a1a2e" opacity={0.5}>
                  <title>{`(${x}, ${y}) Static cell`}</title>
                </rect>
              );
            }
            const kl = cellKL(gtDist, predDist);
            return (
              <rect key={`${y}-${x}`} x={x} y={y} width={1} height={1} fill={divergenceColor(kl)}>
                <title>{`(${x}, ${y}) KL: ${kl.toFixed(4)}`}</title>
              </rect>
            );
          }),
        )}
      </svg>
      <div className="divergence-scale">
        <span>Good</span>
        <div className="divergence-bar" />
        <span>Poor</span>
      </div>
    </div>
  );
}

function SeedPredictionPanel(props: {
  seedIdx: number;
  analysis: SeedAnalysis;
  initialState: InitialState;
  viewMode: "blend" | "argmax";
}) {
  const hasPred = props.analysis.prediction !== null;
  const hasGt = props.analysis.groundTruth !== null;

  let klScore: number | null = null;
  let compScore: number | null = null;
  if (hasPred && hasGt) {
    klScore = scorePrediction(props.analysis.prediction!, props.analysis.groundTruth!);
    compScore = competitionScore(klScore);
  }

  return (
    <section className="panel">
      <div className="section-label-row">
        <h2>Seed {props.seedIdx}</h2>
        <span>
          {compScore !== null
            ? `Score: ${compScore.toFixed(1)}/100 · KL: ${klScore!.toFixed(4)}`
            : hasPred
              ? "No ground truth for comparison"
              : hasGt
                ? "No prediction — run montecarlo"
                : "No data available"}
        </span>
      </div>

      <div className="prediction-maps-grid">
        <div>
          <div className="prediction-map-label">Initial state</div>
          <MiniMap grid={props.initialState.grid} settlements={props.initialState.settlements} />
        </div>

        {hasPred && (
          <DistributionMap grid={props.analysis.prediction!} mode={props.viewMode} label="Prediction" />
        )}

        {hasGt && (
          <DistributionMap grid={props.analysis.groundTruth!} mode={props.viewMode} label="Ground truth" />
        )}

        {hasPred && hasGt && (
          <DivergenceMap prediction={props.analysis.prediction!} groundTruth={props.analysis.groundTruth!} />
        )}
      </div>
    </section>
  );
}

// ── Exported page content ────────────────────────────────────────────────────

export function PredictionsContent(props: { round: RoundData }) {
  const [viewMode, setViewMode] = useState<"blend" | "argmax">("blend");
  const { round } = props;
  const analyses = round.seedAnalyses;

  const seedScores = analyses
    .map((a) => {
      if (a.prediction && a.groundTruth) {
        const kl = scorePrediction(a.prediction, a.groundTruth);
        return { kl, score: competitionScore(kl) };
      }
      return null;
    })
    .filter((s): s is { kl: number; score: number } => s !== null);

  const avgScore =
    seedScores.length > 0 ? seedScores.reduce((sum, s) => sum + s.score, 0) / seedScores.length : null;
  const avgKl =
    seedScores.length > 0 ? seedScores.reduce((sum, s) => sum + s.kl, 0) / seedScores.length : null;

  const hasAnyData = analyses.some((a) => a.prediction !== null || a.groundTruth !== null);

  return (
    <>
      <section className="focus-hero">
        <div className="hero-metrics">
          <MetricCard
            label="Seeds"
            value={String(round.seedsCount)}
            hint={`${round.mapWidth}x${round.mapHeight} map`}
          />
          {avgScore !== null ? (
            <MetricCard
              label="Average score"
              value={`${avgScore.toFixed(1)}/100`}
              hint={`KL: ${avgKl!.toFixed(4)}`}
            />
          ) : (
            <MetricCard label="Score" value="—" hint="Needs prediction + ground truth" />
          )}
          <MetricCard
            label="View mode"
            value={viewMode === "blend" ? "Blend" : "Argmax"}
            hint="Color blending strategy"
          />
        </div>
      </section>

      <section className="panel">
        <div className="section-label-row">
          <h2>View mode</h2>
          <span>How to render probability distributions</span>
        </div>
        <div className="action-row">
          <button
            className={viewMode === "blend" ? "snapshot-pill active" : "snapshot-pill"}
            onClick={() => setViewMode("blend")}
            type="button"
          >
            Blend colors
          </button>
          <button
            className={viewMode === "argmax" ? "snapshot-pill active" : "snapshot-pill"}
            onClick={() => setViewMode("argmax")}
            type="button"
          >
            Most likely class
          </button>
        </div>
        <div className="legend prediction-legend">
          {CLASS_NAMES.map((name, i) => {
            const [r, g, b] = CLASS_COLORS[i];
            return (
              <span className="legend-item" key={i}>
                <i style={{ backgroundColor: `rgb(${r},${g},${b})` }} />
                {name}
              </span>
            );
          })}
        </div>
      </section>

      {!hasAnyData && (
        <section className="panel">
          <div className="empty-inline">
            No prediction or ground truth data found for this round.
          </div>
        </section>
      )}

      {analyses.map((analysis, seedIdx) => {
        if (!analysis.prediction && !analysis.groundTruth) return null;
        return (
          <SeedPredictionPanel
            key={seedIdx}
            seedIdx={seedIdx}
            analysis={analysis}
            initialState={round.initialStates[seedIdx]}
            viewMode={viewMode}
          />
        );
      })}
    </>
  );
}
