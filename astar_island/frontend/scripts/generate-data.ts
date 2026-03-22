import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type InitialSettlement = {
  x: number;
  y: number;
  has_port: boolean;
  alive: boolean;
};

type QuerySettlement = InitialSettlement & {
  population: number;
  food: number;
  wealth: number;
  defense: number;
  owner_id: number;
};

type InitialState = {
  grid: number[][];
  settlements: InitialSettlement[];
};

type RoundDetails = {
  id: string;
  round_number: number;
  event_date: string;
  status: string;
  map_width: number;
  map_height: number;
  prediction_window_minutes: number;
  started_at: string;
  closes_at: string;
  round_weight: number;
  seeds_count: number;
  initial_states: InitialState[];
};

type QueryResult = {
  grid: number[][];
  settlements: QuerySettlement[];
  viewport: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  width: number;
  height: number;
  queries_used: number;
  queries_max: number;
};

type QueryFile = {
  id: string;
  fileName: string;
  mapIdx: number;
  row: number;
  col: number;
  windowWidth: number;
  windowHeight: number;
  variantKey: string | null;
  variantValue: number | null;
  grid: number[][];
  settlements: QuerySettlement[];
  viewport: QueryResult["viewport"];
  mapWidth: number;
  mapHeight: number;
  queriesUsed: number;
  queriesMax: number;
};

type GroundTruthFile = {
  width: number;
  height: number;
  initial_grid: number[][];
  ground_truth: number[][][];
  prediction: number[][][] | null;
  score: number | null;
};

type SeedAnalysis = {
  groundTruth: number[][][] | null;
  prediction: number[][][] | null;
};

type RoundPayload = {
  id: string;
  roundNumber: number;
  eventDate: string;
  status: string;
  mapWidth: number;
  mapHeight: number;
  predictionWindowMinutes: number;
  startedAt: string;
  closesAt: string;
  roundWeight: number;
  seedsCount: number;
  initialStates: InitialState[];
  queryFiles: QueryFile[];
  seedAnalyses: SeedAnalysis[];
};

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = resolve(SCRIPT_DIR, "..");
const DATA_ROOT = resolve(FRONTEND_ROOT, "..", "data");
const OUTPUT_FILE = resolve(FRONTEND_ROOT, "public", "generated", "rounds.json");
const GENERATED_ANALYSIS_ROOT = resolve(FRONTEND_ROOT, "public", "generated", "analysis");

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

async function readJsonIfExists<T>(path: string): Promise<T | null> {
  try {
    return await readJson<T>(path);
  } catch {
    return null;
  }
}

function parseQueryStem(stem: string): Record<string, string> {
  const matches = [...stem.matchAll(/([a-z_]+)=([^_]+)/g)].map((match) => [match[1], match[2]]);
  return Object.fromEntries(matches);
}

async function loadQueryFiles(queryDir: string): Promise<QueryFile[]> {
  let entries: string[] = [];

  try {
    entries = (await readdir(queryDir)).filter((entry) => entry.endsWith(".json"));
  } catch {
    return [];
  }

  const queryFiles = await Promise.all(
    entries.map(async (fileName) => {
      const path = join(queryDir, fileName);
      const stem = fileName.replace(/\.json$/, "");
      const params = parseQueryStem(stem);
      const payload = await readJson<QueryResult>(path);
      const variantKey = ["run_seed_idx", "snapshot_seed"].find((key) => key in params) ?? null;
      const variantValue =
        variantKey === null ? null : Number.parseInt(params[variantKey] ?? "", 10);

      return {
        id: stem,
        fileName,
        mapIdx: Number.parseInt(params.map_idx ?? "0", 10),
        row: Number.parseInt(params.r ?? String(payload.viewport.y), 10),
        col: Number.parseInt(params.c ?? String(payload.viewport.x), 10),
        windowWidth: Number.parseInt(params.w ?? String(payload.viewport.w), 10),
        windowHeight: Number.parseInt(params.h ?? String(payload.viewport.h), 10),
        variantKey,
        variantValue,
        grid: payload.grid,
        settlements: payload.settlements,
        viewport: payload.viewport,
        mapWidth: payload.width,
        mapHeight: payload.height,
        queriesUsed: payload.queries_used,
        queriesMax: payload.queries_max,
      } satisfies QueryFile;
    }),
  );

  queryFiles.sort((left, right) => {
    if (left.mapIdx !== right.mapIdx) {
      return left.mapIdx - right.mapIdx;
    }
    if (left.row !== right.row) {
      return left.row - right.row;
    }
    if (left.col !== right.col) {
      return left.col - right.col;
    }
    return (left.variantValue ?? 0) - (right.variantValue ?? 0);
  });

  return queryFiles;
}

async function loadSeedAnalyses(
  roundDir: string,
  seedsCount: number,
): Promise<SeedAnalysis[]> {
  const analysisDir = join(roundDir, "analysis");
  const analyses: SeedAnalysis[] = [];

  for (let i = 0; i < seedsCount; i++) {
    let groundTruth: number[][][] | null = null;
    let prediction: number[][][] | null = null;

    try {
      const gt = await readJson<GroundTruthFile>(
        join(analysisDir, `ground_truth_seed_index=${i}.json`),
      );
      groundTruth = gt.ground_truth;
    } catch {
      // No ground truth file for this seed.
    }

    try {
      prediction = await readJson<number[][][]>(
        join(analysisDir, `prediction_seed_index=${i}.json`),
      );
    } catch {
      // No prediction file for this seed.
    }

    analyses.push({ groundTruth, prediction });
  }

  return analyses;
}

async function loadRounds(): Promise<RoundPayload[]> {
  const entries = await readdir(DATA_ROOT, { withFileTypes: true });
  const roundDirs = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);

  const rounds = await Promise.all(
    roundDirs.map(async (roundId) => {
      const roundDir = join(DATA_ROOT, roundId);
      const details = await readJson<RoundDetails>(join(roundDir, "details.json"));
      const queryFiles = await loadQueryFiles(join(roundDir, "query"));
      const seedAnalyses = await loadSeedAnalyses(roundDir, details.seeds_count);

      return {
        id: details.id,
        roundNumber: details.round_number,
        eventDate: details.event_date,
        status: details.status,
        mapWidth: details.map_width,
        mapHeight: details.map_height,
        predictionWindowMinutes: details.prediction_window_minutes,
        startedAt: details.started_at,
        closesAt: details.closes_at,
        roundWeight: details.round_weight,
        seedsCount: details.seeds_count,
        initialStates: details.initial_states,
        queryFiles,
        seedAnalyses,
      } satisfies RoundPayload;
    }),
  );

  rounds.sort((left, right) => Date.parse(right.startedAt) - Date.parse(left.startedAt));
  return rounds;
}

async function main() {
  await rm(GENERATED_ANALYSIS_ROOT, { recursive: true, force: true });
  const rounds = await loadRounds();
  const payload = {
    generatedAt: new Date().toISOString(),
    rounds,
  };

  await mkdir(dirname(OUTPUT_FILE), { recursive: true });
  await writeFile(OUTPUT_FILE, JSON.stringify(payload));

  console.log(
    `Wrote ${rounds.length} round(s) and ${rounds.reduce((sum, round) => sum + round.queryFiles.length, 0)} query file(s) to ${OUTPUT_FILE}`,
  );
}

main().catch((error: Error) => {
  console.error(error);
  process.exitCode = 1;
});
