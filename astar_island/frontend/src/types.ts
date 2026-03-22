export type InitialSettlement = {
  x: number;
  y: number;
  has_port: boolean;
  alive: boolean;
};

export type QuerySettlement = InitialSettlement & {
  population: number;
  food: number;
  wealth: number;
  defense: number;
  owner_id: number;
};

export type Distribution = number[];
export type DistributionGrid = Distribution[][];

export type InitialState = {
  grid: number[][];
  settlements: InitialSettlement[];
};

export type QueryFile = {
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
  viewport: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  mapWidth: number;
  mapHeight: number;
  queriesUsed: number;
  queriesMax: number;
};

export type DistributionGrid = number[][][];

export type SeedAnalysis = {
  groundTruth: DistributionGrid | null;
  prediction: DistributionGrid | null;
};

export type RoundData = {
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

export type AppData = {
  generatedAt: string;
  rounds: RoundData[];
};

export type GroundTruthAnalysis = {
  prediction: DistributionGrid | null;
  groundTruth: DistributionGrid;
  score: number | null;
  width: number;
  height: number;
  initialGrid: number[][];
};

export type ReplayFrame = {
  step: number;
  grid: number[][];
  settlements: QuerySettlement[];
};

export type ReplayAnalysis = {
  roundId: string;
  seedIndex: number;
  simSeed: number;
  width: number;
  height: number;
  frames: ReplayFrame[];
};
