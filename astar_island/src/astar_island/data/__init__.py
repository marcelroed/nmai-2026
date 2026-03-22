import json
import random
from datetime import date, datetime
from typing import Annotated, Literal, Self, TypeAlias

from matplotlib.pylab import isin
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from astar_island.data.client import (
    get_active_round_id,
    get_round_details,
    get_simulation_result,
    round_data_path,
)

Grid: TypeAlias = list[list[int]]
QueryKey: TypeAlias = tuple[Literal[0, 1, 2, 3, 4], int, int, int]


@dataclass(frozen=True)
class InitialSettlement:
    row: Annotated[int, Field(validation_alias="y")]
    col: Annotated[int, Field(validation_alias="x")]
    has_port: bool
    alive: bool


@dataclass(frozen=True)
class QuerySettlement(InitialSettlement):
    population: float
    food: float
    wealth: float
    defense: float
    owner_id: int


@dataclass(frozen=True)
class InitialState:
    grid: Grid
    settlements: list[InitialSettlement]


@dataclass(frozen=True)
class Viewport:
    row: Annotated[int, Field(validation_alias="y")]
    col: Annotated[int, Field(validation_alias="x")]
    n_rows: Annotated[int, Field(validation_alias="h")]
    n_cols: Annotated[int, Field(validation_alias="w")]


@dataclass(frozen=True)
class QueryResult:
    grid: Grid
    settlements: list[QuerySettlement]
    viewport: Viewport
    n_cols: Annotated[int, Field(validation_alias="width")]
    n_rows: Annotated[int, Field(validation_alias="height")]
    queries_used: int
    queries_max: int


@dataclass(frozen=True)
class RoundDetails:
    id: str
    round_number: int
    event_date: date
    status: str
    n_cols: Annotated[int, Field(validation_alias="map_width")]
    n_rows: Annotated[int, Field(validation_alias="map_height")]
    prediction_window_minutes: int
    started_at: datetime
    closes_at: datetime
    round_weight: float
    seeds_count: int
    initial_states: list[InitialState]


@dataclass(frozen=True)
class QueryData:
    round_id: str
    details: RoundDetails
    queries: dict[QueryKey, QueryResult]

    @classmethod
    def from_raw(cls, *, round_id: str, details: object, queries: object) -> Self:
        return TypeAdapter(cls).validate_python(
            {"round_id": round_id, "details": details, "queries": queries}
        )

    @classmethod
    def build(cls, round_id: str | None = None) -> Self:
        if round_id is None:
            round_id = get_active_round_id()
        else:
            print(f"using override {round_id=}")

        details = get_round_details(round_id)

        queries: dict[QueryKey, dict] = {}
        tot = 0

        if (query_dir := round_data_path(round_id) / "query").exists():
            assert len(list(query_dir.glob("*"))) == 50 or round_id in [
                "2a341ace-0f57-4309-9b89-e59fe0f09179"
            ], f"{round_id}"
            print("Manually loading the 50 queries")
            for run_path in query_dir.glob("*"):
                r = run_path
                r = run_path.name.split("_r=")[1].split("_")[0]
                c = run_path.name.split("_c=")[1].split("_")[0]
                map_idx = run_path.name.split("map_idx=")[1].split("_")[0]
                run_seed_idx = run_path.name.split("_run_seed_idx=")[1].split("_")[0]
                queries[int(map_idx), int(r), int(c), int(run_seed_idx)] = json.loads(
                    run_path.read_text()
                )  # ty:ignore[invalid-assignment]
        else:
            for run_seed_idx in range(50):
                random.seed(hash(round_id) + run_seed_idx)
                map_idx = random.choice(range(5))
                r = random.choice(range(25))
                c = random.choice(range(25))
                # print(map_idx, r, c, run_seed_idx)
                simulation_result = get_simulation_result(
                    round_id, map_idx=map_idx, r=r, c=c, run_seed_idx=run_seed_idx
                )
                queries[map_idx, r, c, run_seed_idx] = simulation_result  # ty:ignore[invalid-assignment]

        print(f"total simulations: {tot}")

        return cls.from_raw(round_id=round_id, details=details, queries=queries)


if __name__ == "__main__":
    print(QueryData.build())
