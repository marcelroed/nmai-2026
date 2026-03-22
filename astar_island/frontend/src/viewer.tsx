import type {
  Distribution,
  DistributionGrid,
  GroundTruthAnalysis,
  InitialState,
  QueryFile,
  QuerySettlement,
  ReplayAnalysis,
  ReplayFrame,
  RoundData,
} from "./types";

export const TERRAIN_COLORS: Record<number, string> = {
  0: "#c2c7d0",
  1: "#ff9c43",
  2: "#2a93a7",
  3: "#912b2f",
  4: "#4a7b37",
  5: "#7b7d87",
  10: "#22476e",
  11: "#d4c08c",
};

export const TERRAIN_NAMES: Record<number, string> = {
  0: "Empty",
  1: "Settlement",
  2: "Port",
  3: "Ruin",
  4: "Forest",
  5: "Mountain",
  10: "Ocean",
  11: "Plains",
};

export type QueryGroup = {
  id: string;
  mapIdx: number;
  row: number;
  col: number;
  windowWidth: number;
  windowHeight: number;
  variantKey: string;
  items: QueryFile[];
};

export type CellCoordinate = {
  row: number;
  col: number;
};

const DISTRIBUTION_CLASS_IDS = [0, 1, 2, 3, 4, 5] as const;
export const ANALYSIS_TERRAIN_IDS = [1, 2, 3, 4, 5, 0] as const;

export function terrainColor(value: number) {
  return TERRAIN_COLORS[value] ?? "#201a28";
}

export function ownerColor(ownerId: number) {
  return `hsl(${(ownerId * 47) % 360} 78% 58%)`;
}

export function formatRoundTime(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

export function variantLabel(key: string) {
  return key === "run_seed_idx" ? "Run seed" : key === "snapshot_seed" ? "Snapshot" : "Query";
}

export function dominantTerrain(distribution: Distribution) {
  let bestIndex = 0;

  for (let index = 1; index < distribution.length; index += 1) {
    if (distribution[index] > distribution[bestIndex]) {
      bestIndex = index;
    }
  }

  return bestIndex;
}

export function dominantConfidence(distribution: Distribution) {
  return distribution[dominantTerrain(distribution)] ?? 0;
}

export function cellDistributionAt(grid: DistributionGrid | null, cell: CellCoordinate) {
  return grid?.[cell.row]?.[cell.col] ?? [0, 0, 0, 0, 0, 0];
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function computeDistributionAgreement(analysis: GroundTruthAnalysis) {
  if (analysis.prediction === null) {
    return null;
  }

  let matchingCells = 0;
  const totalCells = analysis.width * analysis.height;

  for (let row = 0; row < analysis.height; row += 1) {
    for (let col = 0; col < analysis.width; col += 1) {
      const predictionDistribution = analysis.prediction[row]?.[col];
      const groundTruthDistribution = analysis.groundTruth[row]?.[col];

      if (!predictionDistribution || !groundTruthDistribution) {
        continue;
      }

      const prediction = dominantTerrain(predictionDistribution);
      const groundTruth = dominantTerrain(groundTruthDistribution);

      if (prediction === groundTruth) {
        matchingCells += 1;
      }
    }
  }

  return {
    matchingCells,
    totalCells,
    differingCells: totalCells - matchingCells,
  };
}

export function computeCellKlDivergence(groundTruth: Distribution, prediction: Distribution) {
  let total = 0;

  for (let index = 0; index < groundTruth.length; index += 1) {
    const p = groundTruth[index] ?? 0;
    const q = Math.max(prediction[index] ?? 0, 1e-9);

    if (p === 0) {
      continue;
    }

    total += p * Math.log(p / q);
  }

  return total;
}

export function buildQueryGroups(round: RoundData | null, mapIdx: number): QueryGroup[] {
  if (round === null) {
    return [];
  }

  const groups = new Map<string, QueryGroup>();

  for (const query of round.queryFiles) {
    if (query.mapIdx !== mapIdx) {
      continue;
    }

    const key = `${query.mapIdx}:${query.row}:${query.col}:${query.windowWidth}:${query.windowHeight}:${query.variantKey ?? "query"}`;
    const existing = groups.get(key);

    if (existing) {
      existing.items.push(query);
      continue;
    }

    groups.set(key, {
      id: key,
      mapIdx: query.mapIdx,
      row: query.row,
      col: query.col,
      windowWidth: query.windowWidth,
      windowHeight: query.windowHeight,
      variantKey: query.variantKey ?? "query",
      items: [query],
    });
  }

  return [...groups.values()]
    .map((group) => ({
      ...group,
      items: [...group.items].sort(
        (left, right) => (left.variantValue ?? 0) - (right.variantValue ?? 0),
      ),
    }))
    .sort((left, right) => {
      if (left.row !== right.row) {
        return left.row - right.row;
      }

      return left.col - right.col;
    });
}

export function sliceGrid(
  grid: number[][],
  row: number,
  col: number,
  height: number,
  width: number,
) {
  return grid.slice(row, row + height).map((line) => line.slice(col, col + width));
}

export function computeDiffCells(initial: number[][], snapshots: QueryFile[]) {
  if (snapshots.length < 2) {
    return { changedCells: 0, totalCells: initial.length * (initial[0]?.length ?? 0) };
  }

  let changedCells = 0;
  const totalCells = initial.length * (initial[0]?.length ?? 0);

  for (let row = 0; row < initial.length; row += 1) {
    for (let col = 0; col < initial[row].length; col += 1) {
      const values = new Set(snapshots.map((snapshot) => snapshot.grid[row][col]));
      if (values.size > 1) {
        changedCells += 1;
      }
    }
  }

  return { changedCells, totalCells };
}

export function tileUsage(grid: number[][]) {
  const counts = new Map<number, number>();

  for (const row of grid) {
    for (const cell of row) {
      counts.set(cell, (counts.get(cell) ?? 0) + 1);
    }
  }

  return [...counts.entries()].sort((left, right) => right[1] - left[1]);
}

export function computeGridDifference(baseGrid: number[][], nextGrid: number[][]) {
  let changedCells = 0;
  let totalCells = 0;

  for (let row = 0; row < baseGrid.length; row += 1) {
    for (let col = 0; col < (baseGrid[row]?.length ?? 0); col += 1) {
      totalCells += 1;
      if (baseGrid[row][col] !== nextGrid[row]?.[col]) {
        changedCells += 1;
      }
    }
  }

  return { changedCells, totalCells };
}

export function countTerrainCells(grid: number[][], terrainId: number) {
  let count = 0;

  for (const row of grid) {
    for (const cell of row) {
      if (cell === terrainId) {
        count += 1;
      }
    }
  }

  return count;
}

export function expectedTerrainCells(grid: DistributionGrid, terrainId: number) {
  let total = 0;

  for (const row of grid) {
    for (const distribution of row) {
      total += distribution[terrainId] ?? 0;
    }
  }

  return total;
}

export function buildReplayTrends(replay: ReplayAnalysis) {
  const terrainCounts = Object.fromEntries(
    DISTRIBUTION_CLASS_IDS.map((terrainId) => [terrainId, [] as number[]]),
  ) as Record<number, number[]>;
  const steps: number[] = [];
  const settlementCounts: number[] = [];
  const uniqueOwnerCounts: number[] = [];
  const portSettlementCounts: number[] = [];
  const totalPopulation: number[] = [];

  for (const frame of replay.frames) {
    steps.push(frame.step);
    settlementCounts.push(frame.settlements.length);
    uniqueOwnerCounts.push(new Set(frame.settlements.map((settlement) => settlement.owner_id)).size);
    portSettlementCounts.push(frame.settlements.filter((settlement) => settlement.has_port).length);
    totalPopulation.push(
      frame.settlements.reduce((sum, settlement) => sum + settlement.population, 0),
    );

    for (const terrainId of DISTRIBUTION_CLASS_IDS) {
      terrainCounts[terrainId].push(countTerrainCells(frame.grid, terrainId));
    }
  }

  return {
    steps,
    terrainCounts,
    settlementCounts,
    uniqueOwnerCounts,
    portSettlementCounts,
    totalPopulation,
  };
}

export function MetricCard(props: { label: string; value: string; hint: string }) {
  return (
    <article className="metric-card">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
      <small>{props.hint}</small>
    </article>
  );
}

export function TrendChart(props: {
  title: string;
  subtitle: string;
  steps: number[];
  series: Array<{ label: string; values: number[]; color: string }>;
  formatter?: (value: number) => string;
}) {
  const width = 320;
  const height = 148;
  const padding = 16;
  const allValues = props.series.flatMap((series) => series.values);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const valueSpan = Math.max(maxValue - minValue, 1);
  const stepSpan = Math.max(props.steps.length - 1, 1);
  const formatter = props.formatter ?? ((value: number) => Math.round(value).toString());

  function pointX(index: number) {
    return padding + ((width - padding * 2) * index) / stepSpan;
  }

  function pointY(value: number) {
    return height - padding - ((value - minValue) / valueSpan) * (height - padding * 2);
  }

  return (
    <article className="mini-panel trend-card">
      <div className="mini-panel-header">
        <strong>{props.title}</strong>
        <span>{props.subtitle}</span>
      </div>
      <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`}>
        <line
          stroke="rgba(255,255,255,0.14)"
          strokeWidth="1"
          x1={padding}
          x2={padding}
          y1={padding}
          y2={height - padding}
        />
        <line
          stroke="rgba(255,255,255,0.14)"
          strokeWidth="1"
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
        />

        {props.series.map((series) => (
          <polyline
            key={series.label}
            fill="none"
            points={series.values
              .map((value, index) => `${pointX(index)},${pointY(value)}`)
              .join(" ")}
            stroke={series.color}
            strokeWidth="2.2"
          />
        ))}
      </svg>
      <div className="trend-axis">
        <span>Step {props.steps[0] ?? 0}</span>
        <span>
          {formatter(minValue)} to {formatter(maxValue)}
        </span>
        <span>Step {props.steps.at(-1) ?? 0}</span>
      </div>
      <div className="legend trend-legend">
        {props.series.map((series) => (
          <span className="legend-item" key={series.label}>
            <i style={{ backgroundColor: series.color }} />
            {series.label}
          </span>
        ))}
      </div>
    </article>
  );
}

export function MiniMap(props: {
  grid: number[][];
  settlements: Array<{ x: number; y: number; has_port: boolean; owner_id?: number }>;
}) {
  const height = props.grid.length;
  const width = props.grid[0]?.length ?? 0;

  return (
    <svg className="mini-map" viewBox={`0 0 ${width} ${height}`}>
      {props.grid.flatMap((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <rect
            fill={terrainColor(cell)}
            height="1"
            key={`${rowIndex}-${colIndex}`}
            opacity="0.95"
            width="1"
            x={colIndex}
            y={rowIndex}
          />
        )),
      )}
      {props.settlements.map((settlement, index) => (
        <circle
          cx={settlement.x + 0.5}
          cy={settlement.y + 0.5}
          fill={
            settlement.has_port
              ? "#79e1ff"
              : settlement.owner_id
                ? ownerColor(settlement.owner_id)
                : "#f8f3e6"
          }
          key={index}
          opacity="0.95"
          r="0.3"
          stroke="#0f1722"
          strokeWidth="0.1"
        />
      ))}
    </svg>
  );
}

export function WorldMap(props: {
  initialState: InitialState;
  queryGroups: QueryGroup[];
  activeGroupId: string | null;
  selectedSnapshots: QueryFile[];
  overlayOpacity: number;
}) {
  const rows = props.initialState.grid.length;
  const cols = props.initialState.grid[0]?.length ?? 0;

  return (
    <div className="map-frame">
      <svg className="world-map" viewBox={`0 0 ${cols} ${rows}`}>
        {props.initialState.grid.flatMap((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <rect
              fill={terrainColor(cell)}
              height="1"
              key={`base-${rowIndex}-${colIndex}`}
              width="1"
              x={colIndex}
              y={rowIndex}
            />
          )),
        )}

        {props.queryGroups.map((group) => (
          <rect
            fill={group.id === props.activeGroupId ? "rgba(255, 255, 255, 0.06)" : "transparent"}
            height={group.windowHeight}
            key={group.id}
            stroke={group.id === props.activeGroupId ? "#f7d488" : "rgba(255,255,255,0.22)"}
            strokeDasharray={group.id === props.activeGroupId ? "none" : "0.5 0.4"}
            strokeWidth={group.id === props.activeGroupId ? "0.24" : "0.14"}
            width={group.windowWidth}
            x={group.col}
            y={group.row}
          />
        ))}

        {props.selectedSnapshots.map((snapshot, index) => {
          const stroke = ownerColor(index + 1);

          return (
            <g key={snapshot.id}>
              <g opacity={props.overlayOpacity}>
                {snapshot.grid.flatMap((row, rowIndex) =>
                  row.map((cell, colIndex) => (
                    <rect
                      fill={terrainColor(cell)}
                      height="1"
                      key={`${snapshot.id}-${rowIndex}-${colIndex}`}
                      width="1"
                      x={snapshot.col + colIndex}
                      y={snapshot.row + rowIndex}
                    />
                  )),
                )}
              </g>

              <rect
                fill="none"
                height={snapshot.windowHeight}
                stroke={stroke}
                strokeWidth="0.28"
                width={snapshot.windowWidth}
                x={snapshot.col}
                y={snapshot.row}
              />

              {snapshot.settlements.map((settlement, settlementIndex) => (
                <circle
                  cx={settlement.x + 0.5}
                  cy={settlement.y + 0.5}
                  fill={ownerColor(settlement.owner_id)}
                  key={`${snapshot.id}-settlement-${settlementIndex}`}
                  opacity="0.9"
                  r={settlement.has_port ? "0.34" : "0.27"}
                  stroke="#f6efe3"
                  strokeWidth="0.08"
                />
              ))}
            </g>
          );
        })}

        {props.initialState.settlements.map((settlement, index) => (
          <circle
            cx={settlement.x + 0.5}
            cy={settlement.y + 0.5}
            fill="rgba(248, 243, 230, 0.12)"
            key={`initial-${index}`}
            r={settlement.has_port ? "0.34" : "0.22"}
            stroke="rgba(248, 243, 230, 0.85)"
            strokeWidth="0.06"
          />
        ))}
      </svg>
    </div>
  );
}

export function ViewportStage(props: {
  baseGrid: number[][];
  selectedSnapshots: QueryFile[];
  overlayOpacity: number;
}) {
  const height = props.baseGrid.length;
  const width = props.baseGrid[0]?.length ?? 0;

  return (
    <svg className="viewport-map" viewBox={`0 0 ${width} ${height}`}>
      {props.baseGrid.flatMap((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <rect
            fill={terrainColor(cell)}
            height="1"
            key={`viewport-${rowIndex}-${colIndex}`}
            opacity="0.48"
            width="1"
            x={colIndex}
            y={rowIndex}
          />
        )),
      )}

      {props.selectedSnapshots.map((snapshot, index) => (
        <g key={snapshot.id}>
          <g opacity={props.overlayOpacity}>
            {snapshot.grid.flatMap((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <rect
                  fill={terrainColor(cell)}
                  height="1"
                  key={`${snapshot.id}-${rowIndex}-${colIndex}`}
                  width="1"
                  x={colIndex}
                  y={rowIndex}
                />
              )),
            )}
          </g>

          {snapshot.settlements.map((settlement, settlementIndex) => (
            <SettlementPoint
              key={`${snapshot.id}-local-${settlementIndex}`}
              offsetCol={snapshot.col}
              offsetRow={snapshot.row}
              paletteIndex={index}
              settlement={settlement}
            />
          ))}
        </g>
      ))}
    </svg>
  );
}

function SettlementPoint(props: {
  settlement: QuerySettlement;
  offsetRow: number;
  offsetCol: number;
  paletteIndex: number;
}) {
  const localX = props.settlement.x - props.offsetCol + 0.5;
  const localY = props.settlement.y - props.offsetRow + 0.5;

  return (
    <circle
      cx={localX}
      cy={localY}
      fill={ownerColor(props.settlement.owner_id + props.paletteIndex)}
      opacity="0.92"
      r={props.settlement.has_port ? "0.32" : "0.26"}
      stroke="#fdf8ef"
      strokeWidth="0.08"
    />
  );
}

export function DistributionMap(props: {
  grid: DistributionGrid;
  selectedCell: CellCoordinate;
  onSelectCell: (cell: CellCoordinate) => void;
}) {
  const rows = props.grid.length;
  const cols = props.grid[0]?.length ?? 0;

  return (
    <svg className="analysis-map" viewBox={`0 0 ${cols} ${rows}`}>
      {props.grid.flatMap((row, rowIndex) =>
        row.map((distribution, colIndex) => {
          const terrain = dominantTerrain(distribution);
          const confidence = dominantConfidence(distribution);
          const isSelected =
            props.selectedCell.row === rowIndex && props.selectedCell.col === colIndex;

          return (
            <g key={`${rowIndex}-${colIndex}`}>
              <rect
                fill={terrainColor(terrain)}
                height="1"
                opacity={0.25 + confidence * 0.75}
                style={{ cursor: "pointer" }}
                width="1"
                x={colIndex}
                y={rowIndex}
                onClick={() => props.onSelectCell({ row: rowIndex, col: colIndex })}
              />
              {isSelected ? (
                <rect
                  fill="none"
                  height="1"
                  stroke="#f7d488"
                  strokeWidth="0.18"
                  width="1"
                  x={colIndex}
                  y={rowIndex}
                />
              ) : null}
            </g>
          );
        }),
      )}
    </svg>
  );
}

export function TerrainProbabilityMap(props: {
  grid: DistributionGrid;
  terrainId: number;
  onHoverCell?: (cell: { row: number; col: number; mass: number }) => void;
  onLeave?: () => void;
}) {
  const rows = props.grid.length;
  const cols = props.grid[0]?.length ?? 0;

  return (
    <svg className="analysis-map" onMouseLeave={props.onLeave} viewBox={`0 0 ${cols} ${rows}`}>
      <rect fill="rgba(255,255,255,0.04)" height={rows} width={cols} x="0" y="0" />
      {props.grid.flatMap((row, rowIndex) =>
        row.map((distribution, colIndex) => {
          const mass = distribution[props.terrainId] ?? 0;

          return (
            <rect
              fill={terrainColor(props.terrainId)}
              height="1"
              key={`${rowIndex}-${colIndex}`}
              opacity={mass}
              width="1"
              x={colIndex}
              y={rowIndex}
              onMouseEnter={() => props.onHoverCell?.({ row: rowIndex, col: colIndex, mass })}
            >
              <title>
                {`${TERRAIN_NAMES[props.terrainId]} @ ${colIndex},${rowIndex}: ${mass}`}
              </title>
            </rect>
          );
        }),
      )}
    </svg>
  );
}

function terrainShortLabel(terrainId: number) {
  return (
    {
      0: "Emp",
      1: "Set",
      2: "Prt",
      3: "Run",
      4: "For",
      5: "Mtn",
      10: "Sea",
      11: "Pln",
    }[terrainId] ?? String(terrainId)
  );
}

function buildCenteredWindow(center: CellCoordinate, radius: number) {
  const cells: Array<{ row: number; col: number; rowOffset: number; colOffset: number }> = [];

  for (let rowOffset = -radius; rowOffset <= radius; rowOffset += 1) {
    for (let colOffset = -radius; colOffset <= radius; colOffset += 1) {
      cells.push({
        row: center.row + rowOffset,
        col: center.col + colOffset,
        rowOffset,
        colOffset,
      });
    }
  }

  return cells;
}

export function GroundTruthZoomGrid(props: {
  grid: DistributionGrid;
  terrainId: number;
  center: CellCoordinate;
}) {
  const cells = buildCenteredWindow(props.center, 2);

  return (
    <div className="zoom-grid">
      {cells.map((cell) => {
        const mass = props.grid[cell.row]?.[cell.col]?.[props.terrainId];
        const isCenter = cell.rowOffset === 0 && cell.colOffset === 0;

        return (
          <article
            className={isCenter ? "zoom-cell active" : "zoom-cell"}
            key={`${cell.row}:${cell.col}`}
          >
            <div className="zoom-cell-coord">
              {cell.col},{cell.row}
            </div>
            {mass === undefined ? (
              <div className="zoom-cell-empty">Out</div>
            ) : (
              <>
                <div
                  className="zoom-cell-swatch"
                  style={{
                    backgroundColor: terrainColor(props.terrainId),
                    opacity: Math.max(mass, 0.08),
                  }}
                />
                <div className="zoom-cell-value">{mass.toFixed(6)}</div>
              </>
            )}
          </article>
        );
      })}
    </div>
  );
}

export function InitialTerrainZoomGrid(props: {
  grid: number[][];
  terrainId: number;
  center: CellCoordinate;
}) {
  const cells = buildCenteredWindow(props.center, 2);

  return (
    <div className="zoom-grid">
      {cells.map((cell) => {
        const terrain = props.grid[cell.row]?.[cell.col];
        const isCenter = cell.rowOffset === 0 && cell.colOffset === 0;
        const matchesRowTerrain = terrain === props.terrainId;

        return (
          <article
            className={isCenter ? "zoom-cell active" : "zoom-cell"}
            key={`${cell.row}:${cell.col}`}
          >
            <div className="zoom-cell-coord">
              {cell.col},{cell.row}
            </div>
            {terrain === undefined ? (
              <div className="zoom-cell-empty">Out</div>
            ) : (
              <>
                <div
                  className="zoom-cell-swatch"
                  style={{
                    backgroundColor: terrainColor(terrain),
                    opacity: matchesRowTerrain ? 1 : 0.45,
                  }}
                />
                <div className="zoom-cell-label">{terrainShortLabel(terrain)}</div>
              </>
            )}
          </article>
        );
      })}
    </div>
  );
}

export function InitialTerrainMap(props: {
  grid: number[][];
  terrainId: number;
}) {
  const rows = props.grid.length;
  const cols = props.grid[0]?.length ?? 0;

  return (
    <svg className="analysis-map" viewBox={`0 0 ${cols} ${rows}`}>
      {props.grid.flatMap((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <rect
            fill={terrainColor(cell)}
            height="1"
            key={`base-${rowIndex}-${colIndex}`}
            opacity={cell === props.terrainId ? 0.18 : 0.08}
            width="1"
            x={colIndex}
            y={rowIndex}
          />
        )),
      )}
      {props.grid.flatMap((row, rowIndex) =>
        row.map((cell, colIndex) =>
          cell === props.terrainId ? (
            <rect
              fill={terrainColor(props.terrainId)}
              height="1"
              key={`highlight-${rowIndex}-${colIndex}`}
              opacity="0.95"
              width="1"
              x={colIndex}
              y={rowIndex}
            />
          ) : null,
        ),
      )}
    </svg>
  );
}

export function DistributionBars(props: {
  prediction: Distribution;
  groundTruth: Distribution;
  selectedCell: CellCoordinate;
}) {
  return (
    <div className="distribution-bars">
      {DISTRIBUTION_CLASS_IDS.map((terrainId) => {
        const prediction = props.prediction[terrainId] ?? 0;
        const groundTruth = props.groundTruth[terrainId] ?? 0;

        return (
          <article className="distribution-row" key={terrainId}>
            <div className="distribution-row-header">
              <strong>{TERRAIN_NAMES[terrainId]}</strong>
              <span>
                {formatPercent(prediction)} predicted · {formatPercent(groundTruth)} truth
              </span>
            </div>
            <div className="distribution-bar-stack">
              <div className="distribution-bar-track">
                <div
                  className="distribution-bar prediction"
                  style={{
                    backgroundColor: terrainColor(terrainId),
                    width: `${prediction * 100}%`,
                  }}
                />
              </div>
              <div className="distribution-bar-track truth">
                <div
                  className="distribution-bar truth"
                  style={{
                    backgroundColor: terrainColor(terrainId),
                    width: `${groundTruth * 100}%`,
                  }}
                />
              </div>
            </div>
          </article>
        );
      })}

      <div className="distribution-cell-label">
        Cell {props.selectedCell.col},{props.selectedCell.row}
      </div>
    </div>
  );
}

export function ReplayStage(props: { frame: ReplayFrame }) {
  const rows = props.frame.grid.length;
  const cols = props.frame.grid[0]?.length ?? 0;

  return (
    <svg className="world-map" viewBox={`0 0 ${cols} ${rows}`}>
      {props.frame.grid.flatMap((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <rect
            fill={terrainColor(cell)}
            height="1"
            key={`${rowIndex}-${colIndex}`}
            width="1"
            x={colIndex}
            y={rowIndex}
          />
        )),
      )}

      {props.frame.settlements.map((settlement, index) => (
        <circle
          cx={settlement.x + 0.5}
          cy={settlement.y + 0.5}
          fill={settlement.has_port ? "#79e1ff" : ownerColor(settlement.owner_id)}
          key={`${settlement.owner_id}-${settlement.x}-${settlement.y}-${index}`}
          opacity="0.94"
          r={settlement.has_port ? "0.34" : "0.27"}
          stroke="#f8f3e6"
          strokeWidth="0.08"
        />
      ))}
    </svg>
  );
}
