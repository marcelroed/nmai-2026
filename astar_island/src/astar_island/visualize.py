from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from astar_island.query_data import MAP_IDXS, QueryData

DEFAULT_OUTPUT_PATH = Path("query_data_visualization.html")
TERRAIN_INFO: dict[int, dict[str, str]] = {
    -1: {
        "name": "Unknown",
        "class_label": "N/A",
        "description": "Missing or unmapped terrain value",
        "color": "#201b2d",
    },
    0: {
        "name": "Empty",
        "class_label": "0",
        "description": "Generic empty cell",
        "color": "#a6adbb",
    },
    1: {
        "name": "Settlement",
        "class_label": "1",
        "description": "Active Norse settlement",
        "color": "#e68600",
    },
    2: {
        "name": "Port",
        "class_label": "2",
        "description": "Coastal settlement with harbour",
        "color": "#2388a4",
    },
    3: {
        "name": "Ruin",
        "class_label": "3",
        "description": "Collapsed settlement",
        "color": "#9f2424",
    },
    4: {
        "name": "Forest",
        "class_label": "4",
        "description": "Provides food to adjacent settlements",
        "color": "#2f6b2e",
    },
    5: {
        "name": "Mountain",
        "class_label": "5",
        "description": "Impassable terrain",
        "color": "#7b8191",
    },
    10: {
        "name": "Ocean",
        "class_label": "0 (Empty)",
        "description": "Impassable water, borders the map",
        "color": "#274a7d",
    },
    11: {
        "name": "Plains",
        "class_label": "0 (Empty)",
        "description": "Flat land, buildable",
        "color": "#d2c08f",
    },
}
TERRAIN_COLORS = {code: info["color"] for code, info in TERRAIN_INFO.items()}


def _serialize_query_data(query_data: QueryData) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    all_tile_values: set[int] = set()
    total_initial_settlements = 0
    total_query_settlements = 0

    for map_idx in MAP_IDXS:
        map_data = query_data.maps[map_idx]
        initial_grid = map_data.initial.grid
        combined = map_data.combine_snapshots(snapshot_seed=map_data.snapshot_seeds[0])
        query_grid = combined.grid
        all_tile_values.update(cell for row in initial_grid for cell in row)
        all_tile_values.update(cell for row in query_grid for cell in row)
        total_initial_settlements += len(map_data.initial.settlements)
        total_query_settlements += len(combined.settlements)

        changed_tiles = sum(
            1
            for row_idx, row in enumerate(query_grid)
            for col_idx, cell in enumerate(row)
            if initial_grid[row_idx][col_idx] != cell
        )
        owner_ids = sorted({settlement.owner_id for settlement in combined.settlements})

        maps[str(map_idx)] = {
            "initial_grid": initial_grid,
            "query_grid": query_grid,
            "n_rows": combined.n_rows,
            "n_cols": combined.n_cols,
            "queries_used": combined.queries_used,
            "queries_max": combined.queries_max,
            "changed_tiles": changed_tiles,
            "owner_ids": owner_ids,
            "snapshot_count": len(map_data.snapshots),
            "snapshot_seeds": list(map_data.snapshot_seeds),
            "stitched_snapshot_seed": combined.snapshot_seed,
            "initial_settlements": [
                {
                    "row": settlement.row,
                    "col": settlement.col,
                    "has_port": settlement.has_port,
                    "alive": settlement.alive,
                }
                for settlement in map_data.initial.settlements
            ],
            "query_settlements": [
                {
                    "row": settlement.row,
                    "col": settlement.col,
                    "has_port": settlement.has_port,
                    "alive": settlement.alive,
                    "population": settlement.population,
                    "food": settlement.food,
                    "wealth": settlement.wealth,
                    "defense": settlement.defense,
                    "owner_id": settlement.owner_id,
                }
                for settlement in combined.settlements
            ],
            "snapshots": [
                {
                    "snapshot_seed": snapshot.snapshot_seed,
                    "viewport": {
                        "row": snapshot.viewport.row,
                        "col": snapshot.viewport.col,
                        "n_rows": snapshot.viewport.n_rows,
                        "n_cols": snapshot.viewport.n_cols,
                    },
                    "grid": snapshot.grid,
                    "queries_used": snapshot.queries_used,
                    "queries_max": snapshot.queries_max,
                    "settlements": [
                        {
                            "row": settlement.row,
                            "col": settlement.col,
                            "has_port": settlement.has_port,
                            "alive": settlement.alive,
                            "population": settlement.population,
                            "food": settlement.food,
                            "wealth": settlement.wealth,
                            "defense": settlement.defense,
                            "owner_id": settlement.owner_id,
                        }
                        for settlement in snapshot.settlements
                    ],
                }
                for snapshot in map_data.snapshots
            ],
        }

    return {
        "round_id": query_data.round_id,
        "round_number": query_data.round_number,
        "map_order": list(MAP_IDXS),
        "map_count": len(query_data.maps),
        "total_initial_settlements": total_initial_settlements,
        "total_query_settlements": total_query_settlements,
        "terrain_info": {
            str(key): value for key, value in sorted(TERRAIN_INFO.items(), key=lambda item: item[0])
        },
        "terrain_colors": {str(key): value for key, value in TERRAIN_COLORS.items()},
        "tile_values": sorted(all_tile_values),
        "maps": maps,
    }


def _build_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    title = f"Query Data Visualizer - round {payload['round_number']}"
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #0a1620;
        --bg-soft: rgba(16, 34, 49, 0.82);
        --panel: rgba(247, 238, 223, 0.9);
        --panel-strong: rgba(255, 248, 236, 0.97);
        --text: #10202d;
        --text-soft: #415564;
        --line: rgba(16, 32, 45, 0.1);
        --accent: #ff7f51;
        --accent-soft: rgba(255, 127, 81, 0.16);
        --shadow: 0 28px 60px rgba(0, 0, 0, 0.24);
        --radius: 24px;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(54, 157, 137, 0.32), transparent 28%),
          radial-gradient(circle at top right, rgba(255, 159, 28, 0.18), transparent 32%),
          linear-gradient(160deg, #08121a 0%, #103249 40%, #0b1e2a 100%);
      }}

      body::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
        background-size: 32px 32px;
        mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.45));
      }}

      main {{
        width: min(1600px, calc(100vw - 12px));
        margin: 8px auto;
        display: grid;
        gap: 10px;
      }}

      .header-strip {{
        padding: 10px 12px;
        border-radius: calc(var(--radius) - 8px);
        background:
          linear-gradient(135deg, rgba(8, 26, 38, 0.9), rgba(12, 41, 57, 0.78)),
          linear-gradient(145deg, rgba(255, 127, 81, 0.12), rgba(42, 157, 143, 0.08));
        color: #fef4e5;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }}

      .header-row {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
      }}

      .header-title {{
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 8px;
        min-width: 0;
      }}

      h1 {{
        margin: 0;
        font-size: clamp(1rem, 1.4vw, 1.3rem);
        line-height: 1;
        letter-spacing: -0.03em;
      }}

      .header-subtitle {{
        color: rgba(254, 244, 229, 0.64);
        font-size: 0.76rem;
      }}

      .header-stats {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
      }}

      .header-chip,
      .panel {{
        border-radius: var(--radius);
        box-shadow: var(--shadow);
      }}

      .header-chip {{
        padding: 5px 8px;
        display: inline-flex;
        gap: 6px;
        align-items: baseline;
        background: rgba(255, 248, 236, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }}

      .eyebrow {{
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(254, 244, 229, 0.58);
      }}

      .header-value {{
        font-size: 0.84rem;
        font-weight: 700;
      }}

      .panel {{
        background: var(--panel);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.35);
        overflow: hidden;
      }}

      .maps-panel {{
        padding: 12px;
        display: grid;
        gap: 10px;
      }}

      .toolbar {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        justify-content: space-between;
        align-items: center;
      }}

      .button-row,
      .toggle-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}

      button {{
        border: 0;
        border-radius: 999px;
        padding: 7px 10px;
        background: rgba(16, 32, 45, 0.08);
        color: var(--text);
        font: inherit;
        font-size: 0.86rem;
        cursor: pointer;
        transition:
          transform 180ms ease,
          background-color 180ms ease,
          color 180ms ease,
          box-shadow 180ms ease;
      }}

      button:hover {{
        transform: translateY(-1px);
        background: rgba(16, 32, 45, 0.12);
      }}

      button.active {{
        background: var(--accent);
        color: white;
        box-shadow: 0 16px 28px rgba(255, 127, 81, 0.28);
      }}

      .toggle {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(16, 32, 45, 0.06);
        color: var(--text-soft);
        font-size: 0.85rem;
      }}

      .toggle input {{
        accent-color: var(--accent);
      }}

      .map-stage {{
        position: relative;
        padding: 10px;
        border-radius: calc(var(--radius) - 6px);
        background:
          radial-gradient(circle at 20% 20%, rgba(255, 127, 81, 0.12), transparent 32%),
          linear-gradient(160deg, rgba(8, 22, 32, 0.92), rgba(18, 39, 52, 0.98));
        overflow: hidden;
      }}

      .map-stage::after {{
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        border-radius: inherit;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
      }}

      .map-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }}

      .map-pane {{
        display: grid;
        gap: 6px;
        align-content: start;
      }}

      .map-pane-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        color: rgba(254, 244, 229, 0.86);
      }}

      .map-pane-header strong {{
        color: #fff6e8;
        letter-spacing: 0.01em;
      }}

      .map-pane-header span {{
        color: rgba(254, 244, 229, 0.56);
        font-size: 0.76rem;
      }}

      .map-canvas {{
        display: block;
        width: 100%;
        border-radius: 18px;
        background: rgba(7, 17, 25, 0.72);
        box-shadow: 0 22px 46px rgba(0, 0, 0, 0.28);
      }}

      .map-caption {{
        margin-top: 4px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 8px;
        color: rgba(254, 244, 229, 0.78);
        font-size: 0.84rem;
      }}

      .map-caption strong {{
        color: #fff6e8;
      }}

      .info-grid {{
        display: grid;
        grid-template-columns:
          minmax(210px, 1fr)
          minmax(170px, 0.85fr)
          minmax(170px, 0.9fr)
          minmax(260px, 1.2fr);
        gap: 10px;
        align-items: start;
      }}

      .info-span-2 {{
        grid-column: span 2;
      }}

      .stack {{
        padding: 8px;
        display: grid;
        gap: 6px;
        align-content: start;
        min-height: 0;
        background: var(--panel-strong);
      }}

      .stack h2 {{
        margin: 0 0 6px;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
      }}

      .legend-list,
      .owners,
      .settlement-list,
      .detail-list {{
        display: grid;
        gap: 6px;
      }}

      .legend-item,
      .owner-item,
      .settlement-item,
      .detail-item {{
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        padding: 5px 6px;
        border-radius: 10px;
        background: rgba(16, 32, 45, 0.04);
        border: 1px solid var(--line);
      }}

      .legend-item {{
        justify-content: flex-start;
      }}

      .swatch {{
        width: 10px;
        height: 10px;
        border-radius: 2px;
        flex: 0 0 auto;
        border: 1px solid rgba(0, 0, 0, 0.18);
      }}

      .owner-chip {{
        width: 9px;
        height: 9px;
        border-radius: 999px;
        flex: 0 0 auto;
        box-shadow: 0 0 0 1px rgba(16, 32, 45, 0.1);
      }}

      .item-main {{
        display: flex;
        gap: 6px;
        align-items: center;
        min-width: 0;
      }}

      .item-copy {{
        min-width: 0;
      }}

      .item-title {{
        display: block;
        font-weight: 700;
        font-size: 0.74rem;
      }}

      .item-subtitle {{
        display: block;
        color: var(--text-soft);
        font-size: 0.68rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      .item-value {{
        font-variant-numeric: tabular-nums;
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        font-size: 0.68rem;
      }}

      .settlement-list {{
        height: auto;
        max-height: none;
        overflow: visible;
        padding-right: 2px;
      }}

      .detail-list {{
        min-height: 0;
        align-content: start;
      }}

      .detail-columns {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
        align-items: stretch;
      }}

      .detail-column {{
        display: grid;
        grid-auto-rows: min-content;
        gap: 6px;
        align-content: start;
        height: 100%;
        padding: 6px;
        border-radius: 10px;
        background: rgba(16, 32, 45, 0.035);
        border: 1px solid var(--line);
      }}

      .detail-column-title {{
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-soft);
      }}

      .detail-list code {{
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        color: #24324a;
      }}

      .hint {{
        margin: 0;
        color: var(--text-soft);
        font-size: 0.68rem;
      }}

      @media (max-width: 1080px) {{
        .info-grid {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .info-span-2 {{
          grid-column: span 2;
        }}

        .map-grid {{
          grid-template-columns: 1fr;
        }}
      }}

      @media (max-width: 720px) {{
        main {{
          width: min(100vw - 8px, 1460px);
          margin: 4px auto 10px;
        }}

        .header-strip,
        .maps-panel,
        .stack {{
          padding: 10px;
        }}

        .header-row {{
          align-items: flex-start;
        }}

        .header-stats {{
          justify-content: flex-start;
        }}

        .info-grid {{
          grid-template-columns: 1fr;
        }}

        .info-span-2 {{
          grid-column: span 1;
        }}

        .detail-columns {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="header-strip">
        <div class="header-row">
          <div class="header-title">
            <span class="eyebrow">Astar Island</span>
            <h1>Query Data Visualizer</h1>
            <span class="header-subtitle">initial map plus stitched snapshot view</span>
          </div>
          <div class="header-stats" id="header-stats"></div>
        </div>
      </section>

      <section class="panel maps-panel">
        <div class="toolbar">
          <div class="button-row" id="map-buttons"></div>
          <div class="toggle-row">
            <label class="toggle">
              <input id="settlement-toggle" type="checkbox" checked />
              Show settlements
            </label>
          </div>
        </div>

        <div class="map-stage">
          <div class="map-grid">
            <section class="map-pane">
              <div class="map-pane-header">
                <strong>Initial</strong>
                <span>starting state</span>
              </div>
              <canvas class="map-canvas" data-layer="initial"></canvas>
            </section>
            <section class="map-pane">
              <div class="map-pane-header">
                <strong>Query</strong>
                <span>stitched from snapshots</span>
              </div>
              <canvas class="map-canvas" data-layer="query"></canvas>
            </section>
            <section class="map-pane">
              <div class="map-pane-header">
                <strong>Diff</strong>
                <span>left initial, right stitched</span>
              </div>
              <canvas class="map-canvas" data-layer="diff"></canvas>
            </section>
          </div>
          <div class="map-caption">
            <div><strong id="map-title"></strong></div>
            <div id="map-caption-right"></div>
          </div>
        </div>
      </section>

      <section class="info-grid">
        <section class="panel stack">
          <h2>Hovered Cell</h2>
          <div class="detail-list" id="cell-details"></div>
          <p class="hint">Move across the map to inspect a cell. Click to pin or unpin the selection.</p>
        </section>

        <section class="panel stack">
          <h2>Terrain Legend</h2>
          <div class="legend-list" id="terrain-legend"></div>
        </section>

        <section class="panel stack">
          <h2>Owners</h2>
          <div class="owners" id="owner-list"></div>
        </section>

        <section class="panel stack settlements-panel">
          <h2>Settlements</h2>
          <div class="settlement-list" id="settlement-list"></div>
        </section>
      </section>
    </main>

    <script id="query-data" type="application/json">{payload_json}</script>
    <script>
      const data = JSON.parse(document.getElementById("query-data").textContent);
      const mapButtons = document.getElementById("map-buttons");
      const headerStats = document.getElementById("header-stats");
      const terrainLegend = document.getElementById("terrain-legend");
      const ownerList = document.getElementById("owner-list");
      const settlementList = document.getElementById("settlement-list");
      const cellDetails = document.getElementById("cell-details");
      const mapTitle = document.getElementById("map-title");
      const mapCaptionRight = document.getElementById("map-caption-right");
      const settlementToggle = document.getElementById("settlement-toggle");
      const paneConfigs = Array.from(document.querySelectorAll(".map-canvas")).map((canvas) => ({{
        canvas,
        ctx: canvas.getContext("2d"),
        layer: canvas.dataset.layer,
      }}));

      const state = {{
        mapIdx: data.map_order[0],
        hoveredCell: null,
        pinnedCell: null,
        showSettlements: true,
      }};

      function getMap(mapIdx = state.mapIdx) {{
        return data.maps[String(mapIdx)];
      }}

      function fallbackTerrainColor(value) {{
        const hue = ((Number(value) * 47) % 360 + 360) % 360;
        return `hsl(${{hue}} 48% 56%)`;
      }}

      function terrainColor(value) {{
        return data.terrain_colors[String(value)] || fallbackTerrainColor(value);
      }}

      function terrainInfo(value) {{
        return (
          data.terrain_info[String(value)] || {{
            name: `Value ${{value}}`,
            class_label: "?",
            description: "Unmapped terrain value",
            color: terrainColor(value),
          }}
        );
      }}

      function ownerColor(ownerId) {{
        const hue = ((Number(ownerId) * 61) % 360 + 360) % 360;
        return `hsl(${{hue}} 72% 56%)`;
      }}

      function withAlpha(hexColor, alpha) {{
        if (!hexColor.startsWith("#") || hexColor.length !== 7) {{
          return hexColor;
        }}
        const r = Number.parseInt(hexColor.slice(1, 3), 16);
        const g = Number.parseInt(hexColor.slice(3, 5), 16);
        const b = Number.parseInt(hexColor.slice(5, 7), 16);
        return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
      }}

      function formatNumber(value, digits = 3) {{
        return Number(value).toFixed(digits);
      }}

      function formatCompact(value) {{
        return new Intl.NumberFormat("en-US", {{
          notation: "compact",
          maximumFractionDigits: 1,
        }}).format(value);
      }}

      function prepareMaps() {{
        for (const map of Object.values(data.maps)) {{
          map.initialSettlementLookup = Object.fromEntries(
            map.initial_settlements.map((settlement) => [
              `${{settlement.row}},${{settlement.col}}`,
              settlement,
            ]),
          );
          map.querySettlementLookup = Object.fromEntries(
            map.query_settlements.map((settlement) => [
              `${{settlement.row}},${{settlement.col}}`,
              settlement,
            ]),
          );

          const ownerSummary = {{}};
          for (const settlement of map.query_settlements) {{
            const key = String(settlement.owner_id);
            ownerSummary[key] ||= {{
              owner_id: settlement.owner_id,
              count: 0,
              population: 0,
              wealth: 0,
              food: 0,
              defense: 0,
            }};
            ownerSummary[key].count += 1;
            ownerSummary[key].population += settlement.population;
            ownerSummary[key].wealth += settlement.wealth;
            ownerSummary[key].food += settlement.food;
            ownerSummary[key].defense += settlement.defense;
          }}
          map.ownerSummary = Object.values(ownerSummary)
            .map((owner) => ({{
              ...owner,
              avg_food: owner.count ? owner.food / owner.count : 0,
              avg_wealth: owner.count ? owner.wealth / owner.count : 0,
              avg_defense: owner.count ? owner.defense / owner.count : 0,
            }}))
            .sort((a, b) => b.population - a.population);
        }}
      }}

      function renderHeader() {{
        const totalChangedTiles = Object.values(data.maps).reduce(
          (total, map) => total + map.changed_tiles,
          0,
        );
        const totalOwners = new Set(
          Object.values(data.maps).flatMap((map) => map.owner_ids),
        ).size;
        const cards = [
          ["Round", "#{round_number}"],
          ["Round ID", data.round_id.slice(0, 8)],
          ["Maps", data.map_count],
          ["Initial Settlements", data.total_initial_settlements],
          ["Queried Settlements", data.total_query_settlements],
          ["Changed Tiles", formatCompact(totalChangedTiles)],
          ["Owners Seen", totalOwners],
        ];
        headerStats.innerHTML = cards
          .map(
            ([label, value]) => `
              <span class="header-chip">
                <span class="eyebrow">${{label}}</span>
                <span class="header-value">${{value}}</span>
              </span>
            `,
          )
          .join("");
      }}

      function renderMapButtons() {{
        mapButtons.innerHTML = data.map_order
          .map(
            (mapIdx) =>
              `<button class="${{mapIdx === state.mapIdx ? "active" : ""}}" data-map-idx="${{mapIdx}}">Map ${{mapIdx}}</button>`,
          )
          .join("");
        for (const button of mapButtons.querySelectorAll("button")) {{
          button.addEventListener("click", () => {{
            state.mapIdx = Number(button.dataset.mapIdx);
            state.hoveredCell = null;
            state.pinnedCell = null;
            renderAll();
          }});
        }}
      }}

      function renderTerrainLegend(map) {{
        const values = new Set([
          ...map.initial_grid.flat(),
          ...map.query_grid.flat(),
        ]);
        terrainLegend.innerHTML = [...values]
          .sort((a, b) => a - b)
          .map(
            (value) => `
              <div class="legend-item">
                <span class="swatch" style="background:${{terrainColor(value)}}"></span>
                <div class="item-copy">
                  <span class="item-title">${{terrainInfo(value).name}}</span>
                  <span class="item-subtitle">code ${{value}}, class ${{terrainInfo(value).class_label}}: ${{terrainInfo(value).description}}</span>
                </div>
              </div>
            `,
          )
          .join("");
      }}

      function renderOwners(map) {{
        if (!map.ownerSummary.length) {{
          ownerList.innerHTML = '<p class="hint">No query settlements available.</p>';
          return;
        }}

        ownerList.innerHTML = map.ownerSummary
          .map(
            (owner) => `
              <div class="owner-item">
                <div class="item-main">
                  <span class="owner-chip" style="background:${{ownerColor(owner.owner_id)}}"></span>
                  <div class="item-copy">
                    <span class="item-title">Owner ${{owner.owner_id}}</span>
                    <span class="item-subtitle">
                      ${{owner.count}} settlements, avg defense ${{formatNumber(owner.avg_defense)}}
                    </span>
                  </div>
                </div>
                <span class="item-value">pop ${{formatNumber(owner.population)}}</span>
              </div>
            `,
          )
          .join("");
      }}

      function renderSettlements(map) {{
        const settlements = [...map.query_settlements]
          .sort((a, b) => b.population - a.population)
          .map((settlement) => ({{
            ...settlement,
            label: `Owner ${{settlement.owner_id}}`,
            value: `pop ${{formatNumber(settlement.population)}}`,
          }}));

        settlementList.innerHTML = settlements
          .map((settlement) => {{
            const chip = `<span class="owner-chip" style="background:${{ownerColor(settlement.owner_id)}}"></span>`;
            const port = settlement.has_port ? " port" : "";
            return `
              <div class="settlement-item">
                <div class="item-main">
                  ${{chip}}
                  <div class="item-copy">
                    <span class="item-title">${{settlement.label}}</span>
                    <span class="item-subtitle">r${{settlement.row}} c${{settlement.col}}${{port}}</span>
                  </div>
                </div>
                <span class="item-value">${{settlement.value}}</span>
              </div>
            `;
          }})
          .join("");
      }}

      function currentCell() {{
        return state.pinnedCell || state.hoveredCell;
      }}

      function renderCellDetails(map) {{
        const cell = currentCell();
        if (!cell) {{
          cellDetails.innerHTML = '<p class="hint">No cell selected yet.</p>';
          return;
        }}

        const initialValue = map.initial_grid[cell.row][cell.col];
        const queryValue = map.query_grid[cell.row][cell.col];
        const changed = initialValue !== queryValue;
        const initialSettlement = map.initialSettlementLookup[`${{cell.row}},${{cell.col}}`];
        const querySettlement = map.querySettlementLookup[`${{cell.row}},${{cell.col}}`];
        const initialTerrain = terrainInfo(initialValue);
        const queryTerrain = terrainInfo(queryValue);
        const initialDetails = [
          ["Coordinates", `<code>row=${{cell.row}} col=${{cell.col}}</code>`],
          ["Terrain", `${{initialTerrain.name}} <code>(${{initialValue}})</code>`],
          ["Meaning", initialTerrain.description],
          [
            "Settlement",
            initialSettlement
              ? initialSettlement.has_port
                ? "present, port"
                : "present"
              : "none",
          ],
        ];
        const queryDetails = [
          ["Coordinates", `<code>row=${{cell.row}} col=${{cell.col}}</code>`],
          ["Terrain", `${{queryTerrain.name}} <code>(${{queryValue}})</code>`],
          ["Meaning", queryTerrain.description],
          ["Changed", changed ? "yes" : "no"],
        ];

        if (querySettlement) {{
          queryDetails.push([
            "Settlement",
            `owner ${{querySettlement.owner_id}}, pop ${{formatNumber(querySettlement.population)}}`,
          ]);
          queryDetails.push([
            "Economy",
            `food ${{formatNumber(querySettlement.food)}} | wealth ${{formatNumber(querySettlement.wealth)}}`,
          ]);
          queryDetails.push(["Defense", formatNumber(querySettlement.defense)]);
        }} else {{
          queryDetails.push(["Settlement", "none"]);
        }}

        const renderDetailItems = (details) =>
          details
            .map(
              ([label, value]) => `
                <div class="detail-item">
                  <span class="item-subtitle">${{label}}</span>
                  <span class="item-value">${{value}}</span>
                </div>
              `,
            )
            .join("");

        cellDetails.innerHTML = `
          <div class="detail-columns">
            <section class="detail-column">
              <div class="detail-column-title">Initial State</div>
              ${{renderDetailItems(initialDetails)}}
            </section>
            <section class="detail-column">
              <div class="detail-column-title">Query State</div>
              ${{renderDetailItems(queryDetails)}}
            </section>
          </div>
        `;
      }}

      function drawTerrain(ctx, map, cellSize, layer) {{
        for (let row = 0; row < map.n_rows; row += 1) {{
          for (let col = 0; col < map.n_cols; col += 1) {{
            const initialValue = map.initial_grid[row][col];
            const queryValue = map.query_grid[row][col];
            const x = col * cellSize;
            const y = row * cellSize;

            if (layer === "initial") {{
              ctx.fillStyle = terrainColor(initialValue);
              ctx.fillRect(x, y, cellSize, cellSize);
            }} else if (layer === "query") {{
              ctx.fillStyle = terrainColor(queryValue);
              ctx.fillRect(x, y, cellSize, cellSize);
            }} else if (initialValue === queryValue) {{
              ctx.fillStyle = withAlpha(terrainColor(queryValue), 0.72);
              ctx.fillRect(x, y, cellSize, cellSize);
            }} else {{
              ctx.fillStyle = terrainColor(initialValue);
              ctx.fillRect(x, y, cellSize / 2, cellSize);
              ctx.fillStyle = terrainColor(queryValue);
              ctx.fillRect(x + cellSize / 2, y, cellSize / 2, cellSize);
              ctx.strokeStyle = "rgba(255, 244, 230, 0.55)";
              ctx.lineWidth = Math.max(1, cellSize * 0.06);
              ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
            }}
          }}
        }}

        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        for (let row = 0; row <= map.n_rows; row += 1) {{
          const y = Math.round(row * cellSize) + 0.5;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(map.n_cols * cellSize, y);
          ctx.stroke();
        }}
        for (let col = 0; col <= map.n_cols; col += 1) {{
          const x = Math.round(col * cellSize) + 0.5;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, map.n_rows * cellSize);
          ctx.stroke();
        }}
      }}

      function drawSettlements(ctx, map, cellSize, layer) {{
        if (!state.showSettlements) {{
          return;
        }}

        const settlements =
          layer === "initial" ? map.initial_settlements : map.query_settlements;

        for (const settlement of settlements) {{
          const centerX = (settlement.col + 0.5) * cellSize;
          const centerY = (settlement.row + 0.5) * cellSize;
          const radius =
            layer === "initial"
              ? Math.max(3, cellSize * 0.18)
              : Math.max(
                  3,
                  Math.min(
                    cellSize * 0.48,
                    cellSize * (0.14 + Math.log2(1 + settlement.population) * 0.11),
                  ),
                );

          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
          ctx.fillStyle =
            layer === "initial"
              ? "rgba(255, 255, 255, 0.92)"
              : ownerColor(settlement.owner_id);
          ctx.fill();
          ctx.lineWidth = Math.max(1.2, cellSize * 0.07);
          ctx.strokeStyle =
            layer === "initial"
              ? "rgba(16, 32, 45, 0.55)"
              : "rgba(255, 248, 236, 0.86)";
          ctx.stroke();

          if (settlement.has_port) {{
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius + Math.max(1.4, cellSize * 0.08), 0, Math.PI * 2);
            ctx.lineWidth = Math.max(1.2, cellSize * 0.08);
            ctx.strokeStyle = "#f6bd60";
            ctx.stroke();
          }}
        }}
      }}

      function drawSelection(ctx, cellSize) {{
        const cell = currentCell();
        if (!cell) {{
          return;
        }}

        ctx.save();
        ctx.lineWidth = Math.max(2, cellSize * 0.1);
        ctx.strokeStyle = state.pinnedCell ? "#ff7f51" : "#fff5e8";
        ctx.strokeRect(
          cell.col * cellSize + 1,
          cell.row * cellSize + 1,
          cellSize - 2,
          cellSize - 2,
        );
        ctx.restore();
      }}

      function renderCanvases() {{
        const map = getMap();
        const dpr = window.devicePixelRatio || 1;

        for (const pane of paneConfigs) {{
          const width = pane.canvas.parentElement.clientWidth;
          const cellSize = width / map.n_cols;
          const height = cellSize * map.n_rows;

          pane.canvas.width = Math.round(width * dpr);
          pane.canvas.height = Math.round(height * dpr);
          pane.canvas.style.width = `${{width}}px`;
          pane.canvas.style.height = `${{height}}px`;

          pane.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          pane.ctx.clearRect(0, 0, width, height);
          drawTerrain(pane.ctx, map, cellSize, pane.layer);
          drawSettlements(pane.ctx, map, cellSize, pane.layer);
          drawSelection(pane.ctx, cellSize);
        }}
      }}

      function renderMapMeta(map) {{
        mapTitle.textContent = `Map ${{state.mapIdx}}`;
        mapCaptionRight.textContent =
          `${{map.changed_tiles}} changed tiles in stitched view for snapshot seed ${{map.stitched_snapshot_seed}}, ${{map.snapshot_count}} snapshots across ${{map.snapshot_seeds.length}} snapshot seeds, ${{map.query_settlements.length}} queried settlements`;
      }}

      function renderStatic() {{
        const map = getMap();
        renderMapButtons();
        renderTerrainLegend(map);
        renderOwners(map);
        renderSettlements(map);
        renderMapMeta(map);
      }}

      function renderInteraction() {{
        renderCellDetails(getMap());
        renderCanvases();
      }}

      function renderAll() {{
        renderStatic();
        renderInteraction();
      }}

      function canvasCellFromEvent(canvas, event) {{
        const map = getMap();
        const rect = canvas.getBoundingClientRect();
        const col = Math.floor(((event.clientX - rect.left) / rect.width) * map.n_cols);
        const row = Math.floor(((event.clientY - rect.top) / rect.height) * map.n_rows);
        if (row < 0 || row >= map.n_rows || col < 0 || col >= map.n_cols) {{
          return null;
        }}
        return {{ row, col }};
      }}

      for (const pane of paneConfigs) {{
        pane.canvas.addEventListener("mousemove", (event) => {{
          if (state.pinnedCell) {{
            return;
          }}
          const nextCell = canvasCellFromEvent(pane.canvas, event);
          if (
            nextCell?.row === state.hoveredCell?.row &&
            nextCell?.col === state.hoveredCell?.col
          ) {{
            return;
          }}
          state.hoveredCell = nextCell;
          renderInteraction();
        }});

        pane.canvas.addEventListener("mouseleave", () => {{
          if (state.pinnedCell || !state.hoveredCell) {{
            return;
          }}
          state.hoveredCell = null;
          renderInteraction();
        }});

        pane.canvas.addEventListener("click", (event) => {{
          const cell = canvasCellFromEvent(pane.canvas, event);
          if (!cell) {{
            return;
          }}
          if (
            state.pinnedCell &&
            state.pinnedCell.row === cell.row &&
            state.pinnedCell.col === cell.col
          ) {{
            state.pinnedCell = null;
          }} else {{
            state.pinnedCell = cell;
            state.hoveredCell = cell;
          }}
          renderInteraction();
        }});
      }}

      settlementToggle.addEventListener("change", () => {{
        state.showSettlements = settlementToggle.checked;
        renderCanvases();
      }});

      window.addEventListener("resize", renderCanvases);

      prepareMaps();
      renderHeader();
      renderAll();
    </script>
  </body>
</html>
""".format(
        title=escape(title),
        round_number=payload["round_number"],
        payload_json=payload_json,
    )


def write_query_data_visualization(query_data: QueryData) -> Path:
    output_path = DEFAULT_OUTPUT_PATH
    output_path.write_text(
        _build_html(_serialize_query_data(query_data)), encoding="utf-8"
    )
    return output_path


if __name__ == "__main__":
    output_path = write_query_data_visualization(QueryData.build())
    print(f"wrote visualization to {output_path.resolve()}")
