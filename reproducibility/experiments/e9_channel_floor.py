"""Exact finite-scenario evaluation of the CHOIR reporting-channel floor.

The analysis declares a finite grid of linked-sample reporting channels and a
finite set of modal-cell selection transforms. For every resulting population
channel, this script minimizes the channel-floor functional over all latent class
compositions that agree with the reported KABCO shares to their stated rounding
precision. Each fixed-channel problem is solved exactly as a collection of small
linear programs, one for each possible fractional-knapsack boundary cell.

The output is exact for the declared finite scenario set. It is not an endpoint
for a continuous class of reporting channels or selection functions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


K = 5
ALPHA = 0.10
ROUNDING_HALF_WIDTH = 0.00005
NUMERICAL_TOLERANCE = 1e-9

PTILDE = {
    "baseline": np.array([0.8422, 0.0947, 0.0550, 0.0072, 0.0008]),
    "rural_highspeed": np.array([0.7900, 0.0885, 0.0907, 0.0250, 0.0057]),
    "unrestrained": np.array([0.3881, 0.1458, 0.2243, 0.1582, 0.0835]),
    "motorcycle": np.array([0.1439, 0.1839, 0.3661, 0.2496, 0.0565]),
}

ACHIEVED_WIDTH = {
    "baseline": 1.4026,
    "rural_highspeed": 1.9524,
    "unrestrained": 3.5819,
    "motorcycle": 3.5517,
}

GRID = {
    "a_o": np.linspace(0.72, 0.92, 5),
    "a_interior": np.linspace(0.46, 0.58, 5),
    "a_k": np.linspace(0.92, 1.00, 3),
    "down_fraction": np.linspace(0.70, 0.95, 4),
    "leak": np.linspace(0.00, 0.18, 3),
}

SELECTION_INDICES = (1, 2, 5)


@dataclass(frozen=True)
class FloorResult:
    stratum: str
    selection_index: int
    floor: float
    achieved_width: float
    feasible_channels: int
    evaluated_channels: int
    a_o: float
    a_interior: float
    a_k: float
    down_fraction: float
    leak: float
    boundary_y: int
    boundary_t: int
    boundary_fraction: float
    latent_composition: list[float]
    reconstructed_reported_composition: list[float]
    maximum_composition_residual: float


def build_linked_channel(
    a_o: float,
    a_interior: float,
    a_k: float,
    down_fraction: float,
    leak: float,
) -> np.ndarray:
    """Build one linked-sample channel from the declared parameter grid."""
    reliability = [a_o, a_interior, a_interior, a_interior, a_k]
    channel = np.zeros((K, K), dtype=float)
    for y in range(K):
        agreement = reliability[y]
        disagreement = 1.0 - agreement
        down = down_fraction * disagreement
        up = (1.0 - down_fraction) * disagreement
        channel[y, y] = agreement
        channel[y, max(0, y - 1)] += down
        channel[y, min(K - 1, y + 1)] += up
    for y in (0, 1):
        moved = leak * channel[y, y]
        channel[y, y] -= moved
        channel[y, 3] += moved
    return channel / channel.sum(axis=1, keepdims=True)


def apply_selection_scenario(linked_channel: np.ndarray, selection_index: int) -> np.ndarray:
    """Apply the declared modal-cell concentration scenario for one selection index."""
    population_channel = linked_channel.copy()
    for y in range(K):
        weights = np.ones(K, dtype=float)
        weights[int(np.argmax(population_channel[y]))] = float(selection_index)
        population_channel[y] *= weights
    return population_channel / population_channel.sum(axis=1, keepdims=True)


def fractional_knapsack_floor(channel: np.ndarray, latent_composition: np.ndarray) -> float:
    """Evaluate the floor for a fixed channel and fixed latent composition."""
    cells = sorted(
        ((float(channel[y, t]), y, t) for y in range(K) for t in range(K)),
        reverse=True,
    )
    required = 1.0 - ALPHA
    coverage = 0.0
    width = 0.0
    for density, y, _t in cells:
        cell_coverage = latent_composition[y] * density
        cell_width = latent_composition[y]
        if coverage + cell_coverage >= required - NUMERICAL_TOLERANCE:
            if cell_coverage > NUMERICAL_TOLERANCE:
                width += (required - coverage) * cell_width / cell_coverage
            return float(width)
        coverage += cell_coverage
        width += cell_width
    raise RuntimeError("The channel does not supply the required coverage mass.")


def exact_floor_for_channel(
    channel: np.ndarray,
    reported_composition: np.ndarray,
    rounding_half_width: float = ROUNDING_HALF_WIDTH,
) -> dict[str, object] | None:
    """Minimize the floor over every latent composition allowed by rounded shares."""
    lower = np.maximum(0.0, reported_composition - rounding_half_width)
    upper = np.minimum(1.0, reported_composition + rounding_half_width)
    composition_a_ub = np.vstack((channel.T, -channel.T))
    composition_b_ub = np.concatenate((upper, -lower))
    cells = sorted(
        ((float(channel[y, t]), y, t) for y in range(K) for t in range(K)),
        reverse=True,
    )
    required = 1.0 - ALPHA
    best: dict[str, object] | None = None

    for boundary_index, (boundary_density, boundary_y, boundary_t) in enumerate(cells):
        if boundary_density <= NUMERICAL_TOLERANCE:
            continue

        prior_count = np.zeros(K, dtype=float)
        prior_coverage = np.zeros(K, dtype=float)
        for density, y, _t in cells[:boundary_index]:
            prior_count[y] += 1.0
            prior_coverage[y] += density

        through_boundary = prior_coverage.copy()
        through_boundary[boundary_y] += boundary_density
        a_ub = np.vstack(
            (
                composition_a_ub,
                prior_coverage,
                -through_boundary,
            )
        )
        b_ub = np.concatenate(
            (
                composition_b_ub,
                np.array([required, -required]),
            )
        )

        objective = prior_count - prior_coverage / boundary_density
        constant = required / boundary_density
        solution = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=np.ones((1, K), dtype=float),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * K,
            method="highs",
        )
        if not solution.success:
            continue

        latent = np.asarray(solution.x, dtype=float)
        floor = float(solution.fun + constant)
        coverage_before = float(prior_coverage @ latent)
        boundary_mass = float(latent[boundary_y] * boundary_density)
        if boundary_mass <= NUMERICAL_TOLERANCE:
            continue
        boundary_fraction = (required - coverage_before) / boundary_mass
        if boundary_fraction < -1e-7 or boundary_fraction > 1.0 + 1e-7:
            continue

        direct_floor = fractional_knapsack_floor(channel, latent)
        if not np.isclose(floor, direct_floor, atol=2e-7, rtol=0.0):
            raise AssertionError((floor, direct_floor))
        reconstructed = channel.T @ latent
        residual = float(np.max(np.maximum(lower - reconstructed, reconstructed - upper)))
        if residual > 2e-8:
            raise AssertionError(residual)

        candidate = {
            "floor": floor,
            "latent_composition": latent,
            "reconstructed_reported_composition": reconstructed,
            "maximum_composition_residual": max(0.0, residual),
            "boundary_y": boundary_y,
            "boundary_t": boundary_t,
            "boundary_fraction": float(np.clip(boundary_fraction, 0.0, 1.0)),
        }
        if best is None or floor < float(best["floor"]):
            best = candidate

    return best


def declared_parameter_grid():
    """Yield every channel parameter tuple in the declared finite set."""
    yield from product(
        GRID["a_o"],
        GRID["a_interior"],
        GRID["a_k"],
        GRID["down_fraction"],
        GRID["leak"],
    )


def evaluate() -> list[FloorResult]:
    """Evaluate every stratum and selection index."""
    parameters = list(declared_parameter_grid())
    output: list[FloorResult] = []
    for stratum, reported in PTILDE.items():
        for selection_index in SELECTION_INDICES:
            best: tuple[dict[str, object], tuple[float, ...]] | None = None
            feasible_channels = 0
            for parameter_values in parameters:
                linked = build_linked_channel(*parameter_values)
                channel = apply_selection_scenario(linked, selection_index)
                result = exact_floor_for_channel(channel, reported)
                if result is None:
                    continue
                feasible_channels += 1
                if best is None or float(result["floor"]) < float(best[0]["floor"]):
                    best = (result, parameter_values)
            if best is None:
                raise RuntimeError(f"No feasible channel for {stratum}, R={selection_index}.")

            result, parameter_values = best
            output.append(
                FloorResult(
                    stratum=stratum,
                    selection_index=selection_index,
                    floor=float(result["floor"]),
                    achieved_width=ACHIEVED_WIDTH[stratum],
                    feasible_channels=feasible_channels,
                    evaluated_channels=len(parameters),
                    a_o=float(parameter_values[0]),
                    a_interior=float(parameter_values[1]),
                    a_k=float(parameter_values[2]),
                    down_fraction=float(parameter_values[3]),
                    leak=float(parameter_values[4]),
                    boundary_y=int(result["boundary_y"]),
                    boundary_t=int(result["boundary_t"]),
                    boundary_fraction=float(result["boundary_fraction"]),
                    latent_composition=np.asarray(result["latent_composition"]).tolist(),
                    reconstructed_reported_composition=np.asarray(
                        result["reconstructed_reported_composition"]
                    ).tolist(),
                    maximum_composition_residual=float(result["maximum_composition_residual"]),
                )
            )
    return output


def self_test() -> None:
    """Check analytic cases and agreement with direct fractional knapsack."""
    uniform_composition = np.full(K, 1.0 / K)
    identity = np.eye(K)
    identity_result = exact_floor_for_channel(identity, uniform_composition, 1e-10)
    assert identity_result is not None
    assert np.isclose(identity_result["floor"], 1.0 - ALPHA, atol=1e-8)

    uniform_channel = np.full((K, K), 1.0 / K)
    uniform_result = exact_floor_for_channel(uniform_channel, uniform_composition, 1e-10)
    assert uniform_result is not None
    assert np.isclose(uniform_result["floor"], K * (1.0 - ALPHA), atol=1e-8)

    test_channel = build_linked_channel(0.82, 0.52, 0.96, 0.80, 0.09)
    test_latent = np.array([0.60, 0.15, 0.12, 0.08, 0.05])
    test_reported = test_channel.T @ test_latent
    test_result = exact_floor_for_channel(test_channel, test_reported, 1e-10)
    assert test_result is not None
    expected = fractional_knapsack_floor(test_channel, test_latent)
    assert np.isclose(test_result["floor"], expected, atol=1e-8)


def write_results(results: list[FloorResult], output_dir: Path) -> None:
    """Write machine-readable audit outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e9_channel_floor_audit.json"
    csv_path = output_dir / "e9_channel_floor.csv"
    payload = {
        "method": "exact_linear_programs_over_declared_finite_channel_set",
        "alpha": ALPHA,
        "rounding_half_width": ROUNDING_HALF_WIDTH,
        "selection_indices": list(SELECTION_INDICES),
        "grid_size": int(np.prod([len(values) for values in GRID.values()])),
        "scope": (
            "Exact for the declared finite scenario set and reported-share rounding intervals. "
            "Not an endpoint for a continuous channel or selection-function class."
        ),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    scalar_fields = [
        "stratum",
        "selection_index",
        "floor",
        "achieved_width",
        "feasible_channels",
        "evaluated_channels",
        "a_o",
        "a_interior",
        "a_k",
        "down_fraction",
        "leak",
        "boundary_y",
        "boundary_t",
        "boundary_fraction",
        "maximum_composition_residual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in scalar_fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.self_test_only:
        print("channel-floor self-tests passed")
        return

    results = evaluate()
    write_results(results, args.output_dir)
    for result in results:
        print(
            f"{result.stratum:16s} R={result.selection_index} "
            f"floor={result.floor:.6f} feasible={result.feasible_channels}/"
            f"{result.evaluated_channels}"
        )


if __name__ == "__main__":
    main()
