import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DATA_ROOT = resolve(SCRIPT_DIR, "..", "..", "data");

async function main() {
  const roundId = process.argv[2];
  if (!roundId) {
    console.error("Usage: bun scripts/save-prediction.ts <round_id>");
    console.error("");
    console.error("Pipe montecarlo output to stdin:");
    console.error(
      "  cd simulator && cargo run --release -- montecarlo <round_id> 100 \\",
    );
    console.error("    | bun ../frontend/scripts/save-prediction.ts <round_id>");
    process.exit(1);
  }

  const analysisDir = join(DATA_ROOT, roundId, "analysis");
  await mkdir(analysisDir, { recursive: true });

  const stdin = await readFile("/dev/stdin", "utf8");
  const lines = stdin.split("\n").filter((line) => line.startsWith("SEED_"));

  if (lines.length === 0) {
    console.error("No SEED_N lines found in stdin.");
    process.exit(1);
  }

  let count = 0;
  for (const line of lines) {
    const match = line.match(/^SEED_(\d+): (.+)$/);
    if (!match) continue;
    const seedIndex = match[1];
    const grid = JSON.parse(match[2]);
    const outPath = join(analysisDir, `prediction_seed_index=${seedIndex}.json`);
    await writeFile(outPath, JSON.stringify(grid));
    console.log(`Saved seed ${seedIndex} prediction → ${outPath}`);
    count++;
  }

  console.log(
    `\nSaved ${count} prediction(s). Run \`bun run sync-data\` to update the frontend.`,
  );
}

main().catch((error: Error) => {
  console.error(error);
  process.exitCode = 1;
});
