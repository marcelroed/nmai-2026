import {
  Link,
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useNavigate,
} from "@tanstack/react-router";
import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { useAppData, useSeedAnalysis } from "./data";
import {
  ANALYSIS_TERRAIN_IDS,
  GroundTruthZoomGrid,
  InitialTerrainMap,
  InitialTerrainZoomGrid,
  MiniMap,
  MetricCard,
  ReplayStage,
  TERRAIN_NAMES,
  TERRAIN_COLORS,
  TerrainProbabilityMap,
  TrendChart,
  WorldMap,
  ViewportStage,
  buildQueryGroups,
  buildReplayTrends,
  computeGridDifference,
  computeDiffCells,
  countTerrainCells,
  expectedTerrainCells,
  formatRoundTime,
  formatPercent,
  sliceGrid,
  terrainColor,
  tileUsage,
  variantLabel,
  type QueryGroup,
} from "./viewer";
import { PredictionsContent } from "./predictions";

function RootLayout() {
  return <Outlet />;
}

function LoadingScreen() {
  return (
    <main className="center-shell">
      <section className="empty-state">
        <h1>Loading downloaded rounds</h1>
        <p>The frontend is waiting for the generated dataset from local `data/`.</p>
      </section>
    </main>
  );
}

function ErrorScreen(props: { message: string }) {
  return (
    <main className="center-shell">
      <section className="empty-state">
        <h1>Data failed to load</h1>
        <p>{props.message}</p>
        <p>Run `bun run sync-data` in `frontend/`, then reload the page.</p>
      </section>
    </main>
  );
}

function SelectionPage() {
  const { data, error, isLoading } = useAppData();
  const [selectedRoundId, setSelectedRoundId] = useState<string | null>(null);

  useEffect(() => {
    if (data && data.rounds.length > 0 && selectedRoundId === null) {
      setSelectedRoundId(data.rounds[0].id);
    }
  }, [data, selectedRoundId]);

  if (error) {
    return <ErrorScreen message={error} />;
  }

  if (isLoading || data === null) {
    return <LoadingScreen />;
  }

  const rounds = data.rounds;
  const selectedRound = rounds.find((round) => round.id === selectedRoundId) ?? rounds[0];

  return (
    <main className="selector-shell">
      <aside className="selector-sidebar">
        <p className="eyebrow">Astar Island</p>
        <h1>Choose a round</h1>
        <p className="muted">
          Rounds are ordered by `started_at`. Pick the round first, then lock onto a single map
          seed.
        </p>

        <div className="round-list">
          {rounds.map((round) => {
            const isActive = round.id === selectedRound.id;
            return (
              <button
                key={round.id}
                className={isActive ? "round-card active" : "round-card"}
                onClick={() => setSelectedRoundId(round.id)}
                type="button"
              >
                <strong>Round {round.roundNumber}</strong>
                <span>{formatRoundTime(round.startedAt)}</span>
                <small>
                  {round.queryFiles.length} query files, {round.seedsCount} seeds
                </small>
              </button>
            );
          })}
        </div>
      </aside>

      <section className="selector-main">
        <header className="selector-hero">
          <div>
            <p className="eyebrow">Round {selectedRound.roundNumber}</p>
            <h2>Select a map seed</h2>
            <p className="muted">
              After you pick a seed, the app switches to a focused detail route with just a small
              back button for navigation.
            </p>
          </div>
          <div className="hero-metrics">
            <MetricCard
              label="Created"
              value={formatRoundTime(selectedRound.startedAt)}
              hint={selectedRound.status}
            />
            <MetricCard
              label="Seeds"
              value={String(selectedRound.initialStates.length)}
              hint={`${selectedRound.mapWidth}x${selectedRound.mapHeight} board`}
            />
            <MetricCard
              label="Query files"
              value={String(selectedRound.queryFiles.length)}
              hint="Downloaded locally"
            />
          </div>
          <Link
            className="back-link"
            params={{ roundId: selectedRound.id }}
            to="/round/$roundId/predictions"
          >
            View predictions
          </Link>
        </header>

        <section className="seed-selector-grid">
          {selectedRound.initialStates.map((state, index) => {
            const queryCount = selectedRound.queryFiles.filter((query) => query.mapIdx === index).length;

            return (
              <Link
                className="seed-card"
                key={index}
                params={{ roundId: selectedRound.id, seedIdx: String(index) }}
                to="/round/$roundId/seed/$seedIdx"
              >
                <div className="seed-card-header">
                  <strong>Seed {index}</strong>
                  <span>{queryCount} query files</span>
                </div>
                <MiniMap grid={state.grid} settlements={state.settlements} />
                <div className="seed-card-footer">
                  <span>{state.settlements.length} initial settlements</span>
                  <small>Open focused viewer</small>
                </div>
              </Link>
            );
          })}
        </section>
      </section>
    </main>
  );
}

function FocusPage() {
  const { data, error, isLoading } = useAppData();
  const params = focusRoute.useParams();
  const navigate = useNavigate();
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [selectedSnapshotIds, setSelectedSnapshotIds] = useState<string[]>([]);
  const [overlayOpacity, setOverlayOpacity] = useState(0.58);
  const [replayFrameIndex, setReplayFrameIndex] = useState(0);
  const [hoveredGroundTruthMass, setHoveredGroundTruthMass] = useState<
    Partial<Record<number, { row: number; col: number; mass: number } | null>>
  >({});

  const seedIdx = Number.parseInt(params.seedIdx, 10);
  const selectedRound = data?.rounds.find((round) => round.id === params.roundId) ?? null;
  const selectedInitialState = selectedRound?.initialStates[seedIdx] ?? null;
  const {
    groundTruth: selectedGroundTruth,
    replay: selectedReplay,
    error: analysisError,
    isLoading: isAnalysisLoading,
  } = useSeedAnalysis(selectedRound?.id ?? null, seedIdx);
  const queryGroups = buildQueryGroups(selectedRound, seedIdx);
  const selectedGroup =
    queryGroups.find((group) => group.id === selectedGroupId) ?? queryGroups[0] ?? null;
  const selectedSnapshots =
    selectedGroup?.items.filter((item) => selectedSnapshotIds.includes(item.id)) ?? [];

  useEffect(() => {
    if (selectedGroup === null) {
      setSelectedGroupId(null);
      setSelectedSnapshotIds([]);
      return;
    }

    if (selectedGroup.id !== selectedGroupId) {
      setSelectedGroupId(selectedGroup.id);
    }

    const validIds = new Set(selectedGroup.items.map((item) => item.id));
    const nextSelected = selectedSnapshotIds.filter((id) => validIds.has(id));

    if (nextSelected.length === 0) {
      setSelectedSnapshotIds(selectedGroup.items.slice(0, 3).map((item) => item.id));
      return;
    }

    if (nextSelected.length !== selectedSnapshotIds.length) {
      setSelectedSnapshotIds(nextSelected);
    }
  }, [selectedGroup, selectedGroupId, selectedSnapshotIds]);

  useEffect(() => {
    if (selectedReplay === null) {
      setReplayFrameIndex(0);
      return;
    }

    setReplayFrameIndex(selectedReplay.frames.length - 1);
  }, [selectedReplay, selectedRound?.id, seedIdx]);

  const timelineRounds = [...(data?.rounds ?? [])]
    .filter((round) => round.initialStates[seedIdx] !== undefined)
    .sort((left, right) => Date.parse(left.startedAt) - Date.parse(right.startedAt));

  const currentViewportInitial =
    selectedGroup === null || selectedInitialState === null
      ? null
      : sliceGrid(
          selectedInitialState.grid,
          selectedGroup.row,
          selectedGroup.col,
          selectedGroup.windowHeight,
          selectedGroup.windowWidth,
        );

  const diffStats =
    currentViewportInitial && selectedSnapshots.length > 0
      ? computeDiffCells(currentViewportInitial, selectedSnapshots)
      : { changedCells: 0, totalCells: 0 };
  const replayFrames = Array.isArray(selectedReplay?.frames) ? selectedReplay.frames : [];
  const replayTrends =
    selectedReplay === null || replayFrames.length === 0
      ? null
      : buildReplayTrends({ ...selectedReplay, frames: replayFrames });
  const replayFrame =
    replayFrames[Math.max(0, Math.min(replayFrameIndex, Math.max(replayFrames.length - 1, 0)))] ??
    null;
  const replayDiffFromInitial =
    replayFrame === null || selectedInitialState === null
      ? { changedCells: 0, totalCells: 0 }
      : computeGridDifference(selectedInitialState.grid, replayFrame.grid);
  const replayDiffFromPrevious =
    replayFrame === null || replayFrames.length === 0 || replayFrameIndex === 0
      ? { changedCells: 0, totalCells: replayDiffFromInitial.totalCells }
      : computeGridDifference(replayFrames[replayFrameIndex - 1].grid, replayFrame.grid);
  const replayPreviewFrames =
    replayFrames.length === 0
      ? []
      : replayFrames.slice(
          Math.max(0, replayFrameIndex - 2),
          Math.min(replayFrames.length, replayFrameIndex + 3),
        );

  useEffect(() => {
    if (data === null || selectedRound === null || selectedInitialState === null) {
      return;
    }

    const currentRound = selectedRound;
    const roundsForSeed = [...data.rounds]
      .filter((round) => round.initialStates[seedIdx] !== undefined)
      .sort((left, right) => Date.parse(left.startedAt) - Date.parse(right.startedAt));
    const currentRoundIndex = roundsForSeed.findIndex((round) => round.id === currentRound.id);

    function navigateTo(roundId: string, nextSeedIdx: number) {
      navigate({
        to: "/round/$roundId/seed/$seedIdx",
        params: { roundId, seedIdx: String(nextSeedIdx) },
      });
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }

      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target instanceof HTMLInputElement ||
          target instanceof HTMLTextAreaElement ||
          target instanceof HTMLSelectElement)
      ) {
        return;
      }

      if (event.key === "h" && seedIdx > 0) {
        event.preventDefault();
        navigateTo(currentRound.id, seedIdx - 1);
        return;
      }

      if (event.key === "l" && seedIdx < currentRound.initialStates.length - 1) {
        event.preventDefault();
        navigateTo(currentRound.id, seedIdx + 1);
        return;
      }

      if (event.key === "j" && currentRoundIndex > 0) {
        event.preventDefault();
        navigateTo(roundsForSeed[currentRoundIndex - 1].id, seedIdx);
        return;
      }

      if (event.key === "k" && currentRoundIndex >= 0 && currentRoundIndex < roundsForSeed.length - 1) {
        event.preventDefault();
        navigateTo(roundsForSeed[currentRoundIndex + 1].id, seedIdx);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [data, navigate, seedIdx, selectedInitialState, selectedRound]);

  if (error) {
    return <ErrorScreen message={error} />;
  }

  if (isLoading || data === null) {
    return <LoadingScreen />;
  }

  if (selectedRound === null || selectedInitialState === null || Number.isNaN(seedIdx)) {
    return (
      <main className="center-shell">
        <section className="empty-state">
          <h1>Round or seed not found</h1>
          <p>The selected route does not match the downloaded data on disk.</p>
          <Link className="back-link inline" to="/">
            Back to selector
          </Link>
        </section>
      </main>
    );
  }

  return (
    <main className="focus-shell">
      <header className="focus-header">
        <Link className="back-link" to="/">
          Back
        </Link>
        <Link
          className="back-link"
          params={{ roundId: selectedRound.id }}
          to="/round/$roundId/predictions"
        >
          Predictions
        </Link>
        <div>
          <p className="eyebrow">
            Round {selectedRound.roundNumber} · Seed {seedIdx}
          </p>
          <h1>Focused map seed viewer</h1>
          <p className="muted">
            This view removes the round and seed selectors so the chosen map stays in focus. Use
            `j/k` for rounds and `h/l` for seeds.
          </p>
        </div>
      </header>

      <section className="focus-hero">
        <div className="hero-metrics">
          <MetricCard
            label="Initial settlements"
            value={String(selectedInitialState.settlements.length)}
            hint={`${selectedRound.mapWidth}x${selectedRound.mapHeight} map`}
          />
          <MetricCard
            label="Query groups"
            value={String(queryGroups.length)}
            hint={queryGroups.length > 0 ? "Grouped by viewport" : "No snapshots for this seed"}
          />
          <MetricCard
            label="Query files"
            value={String(selectedRound.queryFiles.filter((query) => query.mapIdx === seedIdx).length)}
            hint="Downloaded for this seed"
          />
          <MetricCard
            label="Analysis"
            value={`${selectedRound.groundTruthCount}/${selectedRound.replayCount}`}
            hint="Ground truth / replay seeds"
          />
        </div>
      </section>

      <section className="panel seed-jump-panel">
        <div className="section-label-row">
          <h2>Switch seeds</h2>
          <span>{selectedRound.initialStates.length} available</span>
        </div>
        <div className="seed-jump-grid">
          {selectedRound.initialStates.map((state, index) => {
            const queryCount = selectedRound.queryFiles.filter((query) => query.mapIdx === index).length;
            const isCurrentSeed = index === seedIdx;

            return (
              <Link
                className={isCurrentSeed ? "seed-jump-card active" : "seed-jump-card"}
                key={index}
                params={{ roundId: selectedRound.id, seedIdx: String(index) }}
                to="/round/$roundId/seed/$seedIdx"
              >
                <div className="seed-card-header">
                  <strong>Seed {index}</strong>
                  <span>{queryCount} queries</span>
                </div>
                <MiniMap grid={state.grid} settlements={state.settlements} />
                <div className="seed-card-footer">
                  <span>{state.settlements.length} settlements</span>
                  <small>{isCurrentSeed ? "Current seed" : "Open seed"}</small>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="panel timeline-panel">
        <div className="section-label-row">
          <h2>Seed evolution across rounds</h2>
          <span>Ordered by creation time</span>
        </div>
        <div className="timeline-row">
          {timelineRounds.map((round) => {
            const state = round.initialStates[seedIdx];
            const isCurrent = round.id === selectedRound.id;

            return (
              <Link
                className={isCurrent ? "timeline-card active" : "timeline-card"}
                key={round.id}
                params={{ roundId: round.id, seedIdx: String(seedIdx) }}
                to="/round/$roundId/seed/$seedIdx"
              >
                <MiniMap grid={state.grid} settlements={state.settlements} />
                <strong>Round {round.roundNumber}</strong>
                <span>{formatRoundTime(round.startedAt)}</span>
                <small>
                  {round.queryFiles.filter((query) => query.mapIdx === seedIdx).length} queries
                </small>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="main-grid">
        <div className="panel">
          <div className="section-label-row">
            <h2>Focused seed map</h2>
            <span>
              {selectedRound.mapWidth}x{selectedRound.mapHeight}
            </span>
          </div>
          <WorldMap
            activeGroupId={selectedGroup?.id ?? null}
            initialState={selectedInitialState}
            overlayOpacity={overlayOpacity}
            queryGroups={queryGroups}
            selectedSnapshots={selectedSnapshots}
          />
          <div className="legend">
            {tileUsage(selectedInitialState.grid).slice(0, 5).map(([value, count]) => (
              <span className="legend-item" key={value}>
                <i style={{ backgroundColor: terrainColor(value) }} />
                {TERRAIN_NAMES[value] ?? `Tile ${value}`} · {count}
              </span>
            ))}
          </div>
        </div>

        <div className="panel control-panel">
          <div className="section-label-row">
            <h2>Snapshot explorer</h2>
            <span>{queryGroups.length} viewport group(s)</span>
          </div>

          {queryGroups.length === 0 ? (
            <p className="muted">
              This seed has no downloaded query windows for the selected round.
            </p>
          ) : (
            <>
              <div className="group-grid">
                {queryGroups.map((group) => (
                  <button
                    className={group.id === selectedGroup?.id ? "group-card active" : "group-card"}
                    key={group.id}
                    onClick={() => {
                      setSelectedGroupId(group.id);
                      setSelectedSnapshotIds(group.items.slice(0, 3).map((item) => item.id));
                    }}
                    type="button"
                  >
                    <strong>
                      {group.row},{group.col}
                    </strong>
                    <span>
                      {group.windowWidth}x{group.windowHeight}
                    </span>
                    <small>
                      {variantLabel(group.variantKey)} · {group.items.length}
                    </small>
                  </button>
                ))}
              </div>

              {selectedGroup && (
                <>
                  <SnapshotLayerSelector
                    selectedGroup={selectedGroup}
                    selectedSnapshotIds={selectedSnapshotIds}
                    setSelectedSnapshotIds={setSelectedSnapshotIds}
                  />

                  <div className="control-subsection">
                    <div className="section-label-row">
                      <h3>Overlay opacity</h3>
                      <span>{Math.round(overlayOpacity * 100)}%</span>
                    </div>
                    <input
                      className="slider"
                      max="0.95"
                      min="0.2"
                      onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                      step="0.05"
                      type="range"
                      value={overlayOpacity}
                    />
                  </div>

                  <div className="stats-strip">
                    <MetricCard
                      label="Viewport"
                      value={`${selectedGroup.windowWidth}x${selectedGroup.windowHeight}`}
                      hint={`Origin ${selectedGroup.row},${selectedGroup.col}`}
                    />
                    <MetricCard
                      label="Cells diverging"
                      value={diffStats.totalCells === 0 ? "0" : `${diffStats.changedCells}`}
                      hint={
                        diffStats.totalCells === 0
                          ? "Select multiple layers"
                          : `${Math.round((diffStats.changedCells / diffStats.totalCells) * 100)}% of viewport`
                      }
                    />
                    <MetricCard
                      label="Query budget"
                      value={
                        selectedSnapshots[0]
                          ? `${selectedSnapshots[0].queriesUsed}/${selectedSnapshots[0].queriesMax}`
                          : "0/0"
                      }
                      hint="Per snapshot file"
                    />
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </section>

      {selectedGroup && currentViewportInitial && (
        <section className="panel">
          <div className="section-label-row">
            <h2>Overlapping viewport preview</h2>
            <span>
              {selectedSnapshots.length > 0
                ? `${selectedSnapshots.length} selected layer(s)`
                : "Choose one or more layers"}
            </span>
          </div>
          <div className="comparison-grid">
            <div className="comparison-stage">
              <ViewportStage
                baseGrid={currentViewportInitial}
                overlayOpacity={overlayOpacity}
                selectedSnapshots={selectedSnapshots}
              />
            </div>

            <div className="small-multiples">
              {selectedSnapshots.length === 0 ? (
                <div className="empty-inline">
                  Select layers to compare the same viewport across overlapping snapshots.
                </div>
              ) : (
                selectedSnapshots.map((snapshot) => (
                  <article className="mini-panel" key={snapshot.id}>
                    <div className="mini-panel-header">
                      <strong>
                        {variantLabel(selectedGroup.variantKey)} {snapshot.variantValue ?? "?"}
                      </strong>
                      <span>{snapshot.settlements.length} settlements</span>
                    </div>
                    <MiniMap
                      grid={snapshot.grid}
                      settlements={snapshot.settlements.map((settlement) => ({
                        ...settlement,
                        x: settlement.x - snapshot.col,
                        y: settlement.y - snapshot.row,
                      }))}
                    />
                  </article>
                ))
              )}
            </div>
          </div>
        </section>
      )}

      <section className="analysis-grid">
        <div className="panel">
          <div className="section-label-row">
            <h2>Ground truth</h2>
            <span>Final distribution after 2.5k steps</span>
          </div>

          {analysisError ? (
            <p className="muted">{analysisError}</p>
          ) : isAnalysisLoading && selectedGroundTruth === null ? (
            <p className="muted">Loading ground truth analysis for this seed.</p>
          ) : selectedGroundTruth === null ? (
            <p className="muted">
              No downloaded ground truth file was found for this round and seed.
            </p>
          ) : (
            <div className="ground-truth-rows">
              <div className="stats-strip">
                <MetricCard
                  label="Score"
                  value={
                    selectedGroundTruth.score === null ? "N/A" : selectedGroundTruth.score.toFixed(4)
                  }
                  hint={
                    selectedGroundTruth.score === null
                      ? "Score unavailable"
                      : "Entropy-weighted KL"
                  }
                />
                <MetricCard
                  label="Rows"
                  value={String(ANALYSIS_TERRAIN_IDS.length)}
                  hint="One row per terrain class"
                />
                <MetricCard
                  label="Map"
                  value={`${selectedGroundTruth.width}x${selectedGroundTruth.height}`}
                  hint="Ground truth grid"
                />
              </div>

              {ANALYSIS_TERRAIN_IDS.map((terrainId) => {
                const expectedCount = expectedTerrainCells(selectedGroundTruth.groundTruth, terrainId);
                const initialCount = countTerrainCells(selectedInitialState.grid, terrainId);
                const hoveredMass = hoveredGroundTruthMass[terrainId] ?? null;

                return (
                  <article className="ground-truth-row" key={terrainId}>
                    <div className="ground-truth-row-header">
                      <div>
                        <strong>{TERRAIN_NAMES[terrainId]}</strong>
                        <span>Ground truth probability vs initial state</span>
                      </div>
                      <div className="ground-truth-row-metrics">
                        <span>Expected {expectedCount.toFixed(1)} cells</span>
                        <span>Initial {initialCount} cells</span>
                        <span>
                          {hoveredMass === null
                            ? "Hover ground truth for exact mass"
                            : `Hover ${hoveredMass.col},${hoveredMass.row} · ${hoveredMass.mass}`}
                        </span>
                      </div>
                    </div>

                    <div className="ground-truth-row-grid plot-section">
                      <article className="mini-panel">
                        <div className="mini-panel-header">
                          <strong>Ground truth</strong>
                          <span>{TERRAIN_NAMES[terrainId]} probability</span>
                        </div>
                        <TerrainProbabilityMap
                          grid={selectedGroundTruth.groundTruth}
                          terrainId={terrainId}
                          onHoverCell={(cell) =>
                            setHoveredGroundTruthMass((current) => ({
                              ...current,
                              [terrainId]: cell,
                            }))
                          }
                          onLeave={() =>
                            setHoveredGroundTruthMass((current) => ({
                              ...current,
                              [terrainId]: null,
                            }))
                          }
                        />
                      </article>

                      <article className="mini-panel">
                        <div className="mini-panel-header">
                          <strong>Initial state</strong>
                          <span>{TERRAIN_NAMES[terrainId]} at step 0</span>
                        </div>
                        <InitialTerrainMap grid={selectedInitialState.grid} terrainId={terrainId} />
                      </article>
                    </div>

                    {hoveredMass ? (
                      <div className="ground-truth-row-grid zoom-section">
                        <article className="mini-panel">
                          <div className="mini-panel-header">
                            <strong>Ground truth 5x5</strong>
                            <span>
                              Center {hoveredMass.col},{hoveredMass.row}
                            </span>
                          </div>
                          <GroundTruthZoomGrid
                            center={{ row: hoveredMass.row, col: hoveredMass.col }}
                            grid={selectedGroundTruth.groundTruth}
                            terrainId={terrainId}
                          />
                        </article>

                        <article className="mini-panel">
                          <div className="mini-panel-header">
                            <strong>Initial state 5x5</strong>
                            <span>
                              Center {hoveredMass.col},{hoveredMass.row}
                            </span>
                          </div>
                          <InitialTerrainZoomGrid
                            center={{ row: hoveredMass.row, col: hoveredMass.col }}
                            grid={selectedInitialState.grid}
                            terrainId={terrainId}
                          />
                        </article>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          )}
        </div>

        <div className="panel replay-panel">
          <div className="section-label-row">
            <h2>Replay</h2>
            <span>One full rollout across all replay snapshots</span>
          </div>

          {analysisError ? (
            <p className="muted">{analysisError}</p>
          ) : isAnalysisLoading && selectedReplay === null ? (
            <p className="muted">Loading replay analysis for this seed.</p>
          ) : selectedReplay === null || replayFrames.length === 0 || replayFrame === null ? (
            <p className="muted">No downloaded replay file was found for this round and seed.</p>
          ) : (
            <>
              {replayTrends && (
                <div className="trend-grid">
                  <TrendChart
                    formatter={(value) => value.toFixed(0)}
                    series={[
                      {
                        label: "Settlement",
                        values: replayTrends.terrainCounts[1],
                        color: TERRAIN_COLORS[1],
                      },
                      {
                        label: "Port",
                        values: replayTrends.terrainCounts[2],
                        color: TERRAIN_COLORS[2],
                      },
                      {
                        label: "Ruin",
                        values: replayTrends.terrainCounts[3],
                        color: TERRAIN_COLORS[3],
                      },
                      {
                        label: "Forest",
                        values: replayTrends.terrainCounts[4],
                        color: TERRAIN_COLORS[4],
                      },
                      {
                        label: "Mountain",
                        values: replayTrends.terrainCounts[5],
                        color: TERRAIN_COLORS[5],
                      },
                    ]}
                    steps={replayTrends.steps}
                    subtitle="Tile counts by type"
                    title="Map composition"
                  />
                  <TrendChart
                    formatter={(value) => value.toFixed(0)}
                    series={[
                      {
                        label: "Total population",
                        values: replayTrends.totalPopulation,
                        color: "#f7d488",
                      },
                      {
                        label: "Settlements",
                        values: replayTrends.settlementCounts,
                        color: "#79e1ff",
                      },
                    ]}
                    steps={replayTrends.steps}
                    subtitle="Population and settlement growth"
                    title="Settler trends"
                  />
                  <TrendChart
                    formatter={(value) => value.toFixed(0)}
                    series={[
                      {
                        label: "Unique owners",
                        values: replayTrends.uniqueOwnerCounts,
                        color: "#9be07d",
                      },
                      {
                        label: "Port settlements",
                        values: replayTrends.portSettlementCounts,
                        color: "#2a93a7",
                      },
                    ]}
                    steps={replayTrends.steps}
                    subtitle="Political spread and harbor count"
                    title="Ownership trends"
                  />
                </div>
              )}

              <div className="action-row replay-actions">
                <button
                  className="ghost-button"
                  onClick={() => setReplayFrameIndex(0)}
                  type="button"
                >
                  Start
                </button>
                <button
                  className="ghost-button"
                  onClick={() => setReplayFrameIndex((current) => Math.max(0, current - 1))}
                  type="button"
                >
                  Prev
                </button>
                <button
                  className="ghost-button"
                  onClick={() =>
                    setReplayFrameIndex((current) => Math.min(replayFrames.length - 1, current + 1))
                  }
                  type="button"
                >
                  Next
                </button>
                <button
                  className="ghost-button"
                  onClick={() => setReplayFrameIndex(replayFrames.length - 1)}
                  type="button"
                >
                  End
                </button>
                <div className="replay-step-label">
                  <strong>
                    Snapshot {replayFrame.step} / {replayFrames.length - 1}
                  </strong>
                  <span>Simulation seed {selectedReplay.simSeed}</span>
                </div>
              </div>

              <input
                className="slider"
                max={replayFrames.length - 1}
                min="0"
                onChange={(event) => setReplayFrameIndex(Number(event.target.value))}
                step="1"
                type="range"
                value={replayFrameIndex}
              />

              <div className="comparison-grid">
                <div className="comparison-stage">
                  <ReplayStage frame={replayFrame} />
                </div>

                <div className="small-multiples">
                  <div className="stats-strip">
                    <MetricCard
                      label="Settlements"
                      value={String(replayFrame.settlements.length)}
                      hint="Alive in this frame"
                    />
                    <MetricCard
                      label="Changed vs start"
                      value={String(replayDiffFromInitial.changedCells)}
                      hint={
                        replayDiffFromInitial.totalCells === 0
                          ? "No cells"
                          : `${formatPercent(
                              replayDiffFromInitial.changedCells / replayDiffFromInitial.totalCells,
                            )} of map`
                      }
                    />
                    <MetricCard
                      label="Changed vs prev"
                      value={String(replayDiffFromPrevious.changedCells)}
                      hint="Per replay frame"
                    />
                  </div>

                  {replayPreviewFrames.map((frame, index) => {
                    const absoluteIndex = Math.max(0, replayFrameIndex - 2) + index;
                    const isActive = absoluteIndex === replayFrameIndex;

                    return (
                      <button
                        className={isActive ? "replay-frame-card active" : "replay-frame-card"}
                        key={frame.step}
                        onClick={() => setReplayFrameIndex(absoluteIndex)}
                        type="button"
                      >
                        <div className="mini-panel-header">
                          <strong>Snapshot {frame.step}</strong>
                          <span>{frame.settlements.length} settlements</span>
                        </div>
                        <MiniMap grid={frame.grid} settlements={frame.settlements} />
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

function SnapshotLayerSelector(props: {
  selectedGroup: QueryGroup;
  selectedSnapshotIds: string[];
  setSelectedSnapshotIds: Dispatch<SetStateAction<string[]>>;
}) {
  return (
    <div className="control-subsection">
      <div className="section-label-row">
        <h3>Layer selection</h3>
        <span>{variantLabel(props.selectedGroup.variantKey)}</span>
      </div>
      <div className="action-row">
        <button
          className="ghost-button"
          onClick={() =>
            props.setSelectedSnapshotIds(props.selectedGroup.items.slice(0, 3).map((item) => item.id))
          }
          type="button"
        >
          Select first 3
        </button>
        <button
          className="ghost-button"
          onClick={() => props.setSelectedSnapshotIds(props.selectedGroup.items.map((item) => item.id))}
          type="button"
        >
          Select all
        </button>
        <button
          className="ghost-button"
          onClick={() => props.setSelectedSnapshotIds([])}
          type="button"
        >
          Clear
        </button>
      </div>
      <div className="snapshot-grid">
        {props.selectedGroup.items.map((item) => {
          const active = props.selectedSnapshotIds.includes(item.id);

          return (
            <button
              className={active ? "snapshot-pill active" : "snapshot-pill"}
              key={item.id}
              onClick={() =>
                props.setSelectedSnapshotIds((current) =>
                  current.includes(item.id)
                    ? current.filter((value) => value !== item.id)
                    : [...current, item.id],
                )
              }
              type="button"
            >
              {variantLabel(props.selectedGroup.variantKey)} {item.variantValue ?? "?"}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PredictionsPage() {
  const { data, error, isLoading } = useAppData();
  const params = predictionsRoute.useParams();

  if (error) {
    return <ErrorScreen message={error} />;
  }

  if (isLoading || data === null) {
    return <LoadingScreen />;
  }

  const selectedRound = data.rounds.find((round) => round.id === params.roundId) ?? null;

  if (selectedRound === null) {
    return (
      <main className="center-shell">
        <section className="empty-state">
          <h1>Round not found</h1>
          <Link className="back-link inline" to="/">
            Back to selector
          </Link>
        </section>
      </main>
    );
  }

  return (
    <main className="focus-shell">
      <header className="focus-header">
        <Link className="back-link" to="/">
          Back
        </Link>
        <div>
          <p className="eyebrow">Round {selectedRound.roundNumber}</p>
          <h1>Predictions</h1>
          <p className="muted">
            Monte Carlo prediction distributions compared with ground truth.
          </p>
        </div>
      </header>
      <PredictionsContent round={selectedRound} />
    </main>
  );
}

const rootRoute = createRootRoute({
  component: RootLayout,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: SelectionPage,
});

const focusRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/round/$roundId/seed/$seedIdx",
  component: FocusPage,
});

const predictionsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/round/$roundId/predictions",
  component: PredictionsPage,
});

const routeTree = rootRoute.addChildren([indexRoute, focusRoute, predictionsRoute]);

export const router = createRouter({
  routeTree,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
