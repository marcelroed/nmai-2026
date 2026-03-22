from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import tty
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from astar_island.data import QueryData, QueryKey, QueryResult
from astar_island.data.client import DATA_DIR
from astar_island.data.client import round_data_path

TerrainCode = Literal[-1, 0, 1, 2, 3, 4, 5, 10, 11]

MAP_IDXS = (0, 1, 2, 3, 4)
DEFAULT_VIEWPORT_SIZE = 15
TERRAIN_META: dict[TerrainCode, dict[str, str]] = {
    -1: {"name": "Unknown", "glyph": "?", "style": "dim", "panel": "dim", "stat": "Unknown"},
    0: {"name": "Empty", "glyph": ".", "style": "bright_black", "panel": "white", "stat": "Empty"},
    1: {"name": "Settlement", "glyph": "S", "style": "yellow", "panel": "yellow", "stat": "Settlement"},
    2: {"name": "Port", "glyph": "P", "style": "bright_cyan", "panel": "bright_cyan", "stat": "Port"},
    3: {"name": "Ruin", "glyph": "R", "style": "red", "panel": "red", "stat": "Ruin"},
    4: {"name": "Forest", "glyph": "♠", "style": "green", "panel": "green", "stat": "Forest"},
    5: {"name": "Mountain", "glyph": "▲", "style": "white", "panel": "white", "stat": "Mountain"},
    10: {"name": "Ocean", "glyph": "~", "style": "blue", "panel": "blue", "stat": "Ocean"},
    11: {"name": "Empty", "glyph": "·", "style": "bright_black", "panel": "white", "stat": "Empty"},
}


@dataclass(frozen=True)
class SeedView:
    map_idx: int
    base_grid: list[list[int]]
    stitched_grid: list[list[int]]
    observed_mask: list[list[bool]]
    queries: list[tuple[QueryKey, QueryResult]]
    settlement_count: int
    alive_settlement_count: int
    port_count: int
    counts: Counter[int]


@dataclass
class AppState:
    query_data: QueryData
    seed_views: dict[int, SeedView]
    cached_round_ids: list[str]
    selected_seed: int = 0
    viewport_row: int = 0
    viewport_col: int = 0
    show_viewport: bool = True
    dim_unobserved: bool = False
    message: str = ""

    @property
    def current_seed(self) -> SeedView:
        return self.seed_views[self.selected_seed]

    @property
    def n_rows(self) -> int:
        return self.query_data.details.n_rows

    @property
    def n_cols(self) -> int:
        return self.query_data.details.n_cols


def _cached_round_id_order() -> list[str]:
    round_rows: list[tuple[int, str, str]] = []
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return []
    for round_dir in data_dir.iterdir():
        if not round_dir.is_dir():
            continue
        details_path = round_dir / "details.json"
        if not details_path.exists():
            continue
        details = json.loads(details_path.read_text())
        round_rows.append(
            (
                int(details.get("round_number", 0)),
                str(details.get("event_date", "")),
                round_dir.name,
            )
        )
    round_rows.sort()
    return [round_id for _, _, round_id in round_rows]


def _load_query_data(round_id: str | None) -> QueryData:
    if round_id is None:
        return QueryData.build()

    details_path = round_data_path(round_id) / "details.json"
    query_dir = round_data_path(round_id) / "query"
    if not details_path.exists():
        raise FileNotFoundError(f"missing cached round details at {details_path}")
    if not query_dir.exists():
        raise FileNotFoundError(f"missing cached queries at {query_dir}")

    details = json.loads(details_path.read_text())
    queries: dict[QueryKey, object] = {}
    for query_path in sorted(query_dir.glob("*.json")):
        name = query_path.name
        map_idx = int(name.split("map_idx=")[1].split("_")[0])
        row = int(name.split("_r=")[1].split("_")[0])
        col = int(name.split("_c=")[1].split("_")[0])
        if "_run_seed_idx=" in name:
            run_idx = int(name.split("_run_seed_idx=")[1].split("_")[0])
        else:
            run_idx = int(name.split("_snapshot_seed=")[1].split("_")[0])
        queries[(map_idx, row, col, run_idx)] = json.loads(query_path.read_text())

    return QueryData.from_raw(round_id=round_id, details=details, queries=queries)


def _build_seed_views(query_data: QueryData) -> dict[int, SeedView]:
    seed_views: dict[int, SeedView] = {}
    for map_idx in MAP_IDXS:
        base_grid = [row[:] for row in query_data.details.initial_states[map_idx].grid]
        stitched_grid = [row[:] for row in base_grid]
        observed_mask = [[False for _ in row] for row in base_grid]

        seed_queries = sorted(
            (
                (key, result)
                for key, result in query_data.queries.items()
                if key[0] == map_idx
            ),
            key=lambda item: (item[0][1], item[0][2], item[0][3]),
        )

        settlements_by_pos: dict[tuple[int, int], object] = {}
        for (_, row, col, _), result in seed_queries:
            for r_off, row_values in enumerate(result.grid):
                for c_off, value in enumerate(row_values):
                    stitched_grid[row + r_off][col + c_off] = value
                    observed_mask[row + r_off][col + c_off] = True
            for settlement in result.settlements:
                settlements_by_pos[(settlement.row, settlement.col)] = settlement

        counts = Counter(cell for grid_row in stitched_grid for cell in grid_row)
        settlement_count = sum(1 for value in stitched_grid for cell in value if cell in (1, 2))
        alive_settlement_count = sum(
            1 for settlement in settlements_by_pos.values() if getattr(settlement, "alive", False)
        )
        port_count = sum(1 for value in stitched_grid for cell in value if cell == 2)

        seed_views[map_idx] = SeedView(
            map_idx=map_idx,
            base_grid=base_grid,
            stitched_grid=stitched_grid,
            observed_mask=observed_mask,
            queries=seed_queries,
            settlement_count=settlement_count,
            alive_settlement_count=alive_settlement_count,
            port_count=port_count,
            counts=counts,
        )

    return seed_views


def _clamp_viewport(state: AppState) -> None:
    state.viewport_row = max(0, min(state.viewport_row, state.n_rows - DEFAULT_VIEWPORT_SIZE))
    state.viewport_col = max(0, min(state.viewport_col, state.n_cols - DEFAULT_VIEWPORT_SIZE))


def _count_viewport(seed_view: SeedView, top: int, left: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in range(top, min(top + DEFAULT_VIEWPORT_SIZE, len(seed_view.stitched_grid))):
        for col in range(left, min(left + DEFAULT_VIEWPORT_SIZE, len(seed_view.stitched_grid[row]))):
            counts[seed_view.stitched_grid[row][col]] += 1
    return counts


def _count_observed(seed_view: SeedView) -> tuple[int, int]:
    observed = sum(1 for row in seed_view.observed_mask for cell in row if cell)
    total = len(seed_view.observed_mask) * len(seed_view.observed_mask[0])
    return observed, total


def _seed_tabs(selected_seed: int) -> Text:
    text = Text()
    for map_idx in MAP_IDXS:
        label = f" Seed {map_idx} "
        style = "bold black on cyan" if map_idx == selected_seed else "bold white on rgb(45,45,70)"
        text.append(label, style=style)
        text.append(" ")
    return text


def _render_grid(state: AppState) -> Text:
    seed_view = state.current_seed
    grid = seed_view.stitched_grid
    row_labels_width = len(str(len(grid) - 1))
    top = state.viewport_row
    left = state.viewport_col
    bottom = min(top + DEFAULT_VIEWPORT_SIZE - 1, len(grid) - 1)
    right = min(left + DEFAULT_VIEWPORT_SIZE - 1, len(grid[0]) - 1)

    text = Text()
    text.append(" " * (row_labels_width + 2))
    for col in range(len(grid[0])):
        if col % 5 == 0:
            text.append(f"{col:<5}", style="dim")
    text.append("\n")

    for row_idx, row in enumerate(grid):
        text.append(f"{row_idx:>{row_labels_width}} ", style="dim")
        for col_idx, cell in enumerate(row):
            meta = TERRAIN_META.get(cell, TERRAIN_META[-1])
            style = meta["style"]
            if state.dim_unobserved and not seed_view.observed_mask[row_idx][col_idx]:
                style = f"{style} dim"
            if state.show_viewport and top <= row_idx <= bottom and left <= col_idx <= right:
                on_edge = row_idx in (top, bottom) or col_idx in (left, right)
                if on_edge:
                    style = f"bold {style} reverse"
                else:
                    style = f"{style} on rgb(25,25,40)"
            text.append(meta["glyph"], style=style)
        text.append(f" {row_idx}", style="dim")
        text.append("\n")

    return text


def _render_sidebar(state: AppState) -> Group:
    seed_view = state.current_seed
    total_observed, total_cells = _count_observed(seed_view)
    viewport_counts = _count_viewport(seed_view, state.viewport_row, state.viewport_col)
    viewport_observed = 0
    for row in range(state.viewport_row, min(state.viewport_row + DEFAULT_VIEWPORT_SIZE, state.n_rows)):
        for col in range(state.viewport_col, min(state.viewport_col + DEFAULT_VIEWPORT_SIZE, state.n_cols)):
            viewport_observed += int(seed_view.observed_mask[row][col])

    legend = Text()
    legend.append("Terrain Legend\n", style="bold yellow")
    for code in (10, 11, 1, 2, 3, 4, 5, -1):
        meta = TERRAIN_META[code]
        legend.append(f"{meta['glyph']} ", style=meta["style"])
        legend.append(f"{meta['name']}\n", style="white")

    summary = Text()
    summary.append(f"Coverage: {100 * total_observed / total_cells:0.0f}%\n", style="bold yellow")
    summary.append(
        f"Viewport: ({state.viewport_row},{state.viewport_col})\n",
        style="bold rgb(255,140,90)",
    )
    summary.append(f"Queries: {len(seed_view.queries)}\n", style="white")
    summary.append(f"Settlement: {seed_view.settlement_count}\n", style="white")
    summary.append(f"Alive settlements: {seed_view.alive_settlement_count}\n", style="white")
    summary.append(f"Port: {seed_view.port_count}\n", style="white")
    summary.append(f"Forest: {seed_view.counts[4]}\n", style="white")
    summary.append(f"Mountain: {seed_view.counts[5]}\n", style="white")
    summary.append(f"Ocean: {seed_view.counts[10]}\n", style="white")

    viewport = Text()
    v_bottom = min(state.viewport_row + DEFAULT_VIEWPORT_SIZE - 1, state.n_rows - 1)
    v_right = min(state.viewport_col + DEFAULT_VIEWPORT_SIZE - 1, state.n_cols - 1)
    viewport.append(
        f"Viewport Contents ({state.viewport_row},{state.viewport_col})->({v_bottom},{v_right})\n",
        style="bold yellow",
    )
    for code in (10, 11, 1, 2, 3, 4, 5):
        count = viewport_counts[code]
        if not count:
            continue
        meta = TERRAIN_META[code]
        viewport.append(f"{meta['glyph']} ", style=meta["style"])
        viewport.append(f"{meta['stat']:<11} {count:>3}\n", style="white")
    viewport.append(
        f"Observed: {viewport_observed}/{DEFAULT_VIEWPORT_SIZE * DEFAULT_VIEWPORT_SIZE}"
        f" ({100 * viewport_observed / (DEFAULT_VIEWPORT_SIZE * DEFAULT_VIEWPORT_SIZE):0.0f}%)\n",
        style="dim",
    )

    return Group(
        Panel(legend, border_style="yellow"),
        Panel(summary, border_style="rgb(255,140,90)"),
        Panel(viewport, border_style="green"),
    )


def _render(state: AppState) -> Panel:
    header = Text()
    header.append("Astar Island Explorer\n", style="bold yellow")
    header.append(
        f"Round #{state.query_data.details.round_number} ({state.query_data.details.status})  "
        f"ID: {state.query_data.round_id}",
        style="dim",
    )

    body = Group(
        header,
        _seed_tabs(state.selected_seed),
        Columns((_render_grid(state), _render_sidebar(state)), expand=False, equal=False),
        Text(
            "1-5 seed   arrows move viewport   v toggle viewport   d dim unobserved   "
            "j prev round   k next round   r refresh   q quit",
            style="dim",
        ),
        Text(state.message, style="italic dim") if state.message else Text(""),
    )
    return Panel(body, border_style="rgb(50,50,70)")


def _read_key(timeout: float | None = None) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    if not select.select([sys.stdin], [], [], 0.01)[0]:
        return first
    second = sys.stdin.read(1)
    if second != "[":
        return first + second
    third = sys.stdin.read(1)
    return f"\x1b[{third}"


def _set_round(state: AppState, round_id: str, *, message: str) -> None:
    refreshed = _load_query_data(round_id)
    state.query_data = refreshed
    state.seed_views = _build_seed_views(refreshed)
    state.selected_seed = min(state.selected_seed, len(MAP_IDXS) - 1)
    _clamp_viewport(state)
    state.message = message


def _move_round(state: AppState, delta: int) -> None:
    if not state.cached_round_ids:
        state.message = "no cached rounds found"
        return
    try:
        current_idx = state.cached_round_ids.index(state.query_data.round_id)
    except ValueError:
        current_idx = 0
    next_idx = max(0, min(current_idx + delta, len(state.cached_round_ids) - 1))
    if next_idx == current_idx:
        state.message = "no more cached rounds in that direction"
        return
    next_round_id = state.cached_round_ids[next_idx]
    _set_round(state, next_round_id, message=f"loaded round {next_round_id}")


def _handle_key(state: AppState, key: str) -> bool:
    if key in {"q", "\x03"}:
        return False
    if key in {"1", "2", "3", "4", "5"}:
        state.selected_seed = int(key) - 1
        state.message = f"selected seed {state.selected_seed}"
        return True
    if key == "v":
        state.show_viewport = not state.show_viewport
        state.message = f"viewport highlight {'on' if state.show_viewport else 'off'}"
        return True
    if key == "d":
        state.dim_unobserved = not state.dim_unobserved
        state.message = f"dim unobserved {'on' if state.dim_unobserved else 'off'}"
        return True
    if key == "j":
        _move_round(state, -1)
        return True
    if key == "k":
        _move_round(state, 1)
        return True
    if key == "r":
        _set_round(state, state.query_data.round_id, message="reloaded query data")
        return True

    if key == "\x1b[A":
        state.viewport_row -= 1
    elif key == "\x1b[B":
        state.viewport_row += 1
    elif key == "\x1b[D":
        state.viewport_col -= 1
    elif key == "\x1b[C":
        state.viewport_col += 1
    else:
        return True

    _clamp_viewport(state)
    state.message = ""
    return True


class _TerminalMode:
    def __enter__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def run_tui(round_id: str | None, once: bool = False) -> None:
    query_data = _load_query_data(round_id)
    state = AppState(
        query_data=query_data,
        seed_views=_build_seed_views(query_data),
        cached_round_ids=_cached_round_id_order(),
    )
    _clamp_viewport(state)
    console = Console()

    if once or not sys.stdin.isatty():
        console.print(_render(state))
        return

    with _TerminalMode(), Live(_render(state), console=console, screen=True, auto_refresh=False) as live:
        live.refresh()
        while True:
            key = _read_key(timeout=None)
            if not _handle_key(state, key):
                break
            live.update(_render(state), refresh=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore Astar Island query data in the terminal.")
    parser.add_argument(
        "--round-id",
        help="Load a cached round from data/<round-id> instead of calling QueryData.build().",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render a single frame and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_tui(round_id=args.round_id, once=args.once)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as exc:
        Console(stderr=True).print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
