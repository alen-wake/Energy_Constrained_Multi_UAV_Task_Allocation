"""Paired experiments, statistical summaries, and English SVG figures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

from algorithm import run_energy_constrained_bundle
from data_scenario import RealDataScenarioGenerator
from experiment_config import (
    BATTERY_CAPACITY_LEVELS,
    DATA_PATH,
    EXECUTION_LOADS,
    FIXED_BATTERY_CAPACITY,
    FIXED_TASK_COMPOSITION,
    FIXED_UAV_COUNT,
    FORMAL_SEEDS,
    GRID_SIZE,
    MAXIMUM_UAV_COUNT,
    METHOD_ORDER,
    REPETITIONS,
    TASK_COMPOSITION_LEVELS,
    TOTAL_TASK_COUNT_LEVELS,
    UAV_COUNT_LEVELS,
)
from model import Scenario
from nearest_feasible_task_baseline import run_nearest_feasible_task_first
from original_consensus_auction_baseline import run_original_consensus_auction
from random_feasible_allocation_baseline import run_random_feasible_allocation


METHOD_COLOURS = {
    "Proposed method": "#1565C0",
    "Original consensus auction": "#6A1B9A",
    "Nearest feasible task first": "#EF6C00",
    "Random feasible allocation": "#C62828",
}
METHOD_MARKERS = {
    "Proposed method": "D",
    "Original consensus auction": "o",
    "Nearest feasible task first": "s",
    "Random feasible allocation": "^",
}
DETAIL_METHODS = tuple(
    method for method in METHOD_ORDER if method != "Random feasible allocation"
)

EXPERIMENT_UAV_COUNT = "UAV count"
EXPERIMENT_BATTERY = "Battery capacity"
EXPERIMENT_TASK_COUNT = "Total task count"
EXPERIMENT_COMPOSITION = "Task composition"


def scenario_sha256(scenario: Scenario) -> str:
    content = {
        "seed": scenario.seed,
        "tasks": [
            [
                task.task_id,
                list(task.coordinate),
                task.task_type,
                task.source_record_id,
            ]
            for task in scenario.tasks
        ],
        "uav_start_positions": [
            list(start) for start in scenario.uav_start_positions
        ],
    }
    encoded = json.dumps(content, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nested_task_scenario(full_scenario: Scenario, total_task_count: int) -> Scenario:
    """Select equal per-type prefixes from the same 200-task scenario."""
    if total_task_count % 4 != 0:
        raise ValueError("A nested task count must be divisible by four")
    count_per_type = total_task_count // 4
    selected_tasks = []
    for task_type in ("I", "M", "S", "N"):
        same_type_tasks = [
            task for task in full_scenario.tasks if task.task_type == task_type
        ]
        if len(same_type_tasks) < count_per_type:
            raise RuntimeError(
                "The full scenario has too few tasks of one type for nesting"
            )
        selected_tasks.extend(same_type_tasks[:count_per_type])
    selected_tasks.sort(key=lambda task: task.task_id)
    return Scenario(
        full_scenario.seed,
        tuple(selected_tasks),
        full_scenario.uav_start_positions,
    )


def run_all_methods(scenario, uav_count, battery_capacity, random_seed):
    """Run every method on the same immutable scenario."""
    method_calls = {
        "Proposed method": lambda: run_energy_constrained_bundle(
            scenario,
            uav_count,
            battery_capacity,
            EXECUTION_LOADS,
            return_details=True,
        ),
        "Original consensus auction": lambda: run_original_consensus_auction(
            scenario,
            uav_count,
            battery_capacity,
            EXECUTION_LOADS,
            return_details=True,
        ),
        "Nearest feasible task first": lambda: run_nearest_feasible_task_first(
            scenario,
            uav_count,
            battery_capacity,
            EXECUTION_LOADS,
            return_details=True,
        ),
        "Random feasible allocation": lambda: run_random_feasible_allocation(
            scenario,
            uav_count,
            battery_capacity,
            EXECUTION_LOADS,
            random_seed + 900_000,
            return_details=True,
        ),
    }
    return {method: method_calls[method]() for method in METHOD_ORDER}


def build_conditions():
    conditions = []
    for uav_count in UAV_COUNT_LEVELS:
        conditions.append(
            (
                EXPERIMENT_UAV_COUNT,
                str(uav_count),
                uav_count,
                FIXED_BATTERY_CAPACITY,
                FIXED_TASK_COMPOSITION,
            )
        )
    for battery_capacity in BATTERY_CAPACITY_LEVELS:
        conditions.append(
            (
                EXPERIMENT_BATTERY,
                str(battery_capacity),
                FIXED_UAV_COUNT,
                battery_capacity,
                FIXED_TASK_COMPOSITION,
            )
        )
    for total_task_count in TOTAL_TASK_COUNT_LEVELS:
        conditions.append(
            (
                EXPERIMENT_TASK_COUNT,
                str(total_task_count),
                FIXED_UAV_COUNT,
                FIXED_BATTERY_CAPACITY,
                None,
            )
        )
    for name, composition in TASK_COMPOSITION_LEVELS.items():
        conditions.append(
            (
                EXPERIMENT_COMPOSITION,
                name,
                FIXED_UAV_COUNT,
                FIXED_BATTERY_CAPACITY,
                composition,
            )
        )
    return conditions


def _task_longitude(task):
    return task.original_coordinate[0] if task.original_coordinate else None


def _task_latitude(task):
    return task.original_coordinate[1] if task.original_coordinate else None


def generate_all_records():
    generator = RealDataScenarioGenerator(DATA_PATH, grid_size=GRID_SIZE)
    result_records = []
    scenario_records = []
    conditions = build_conditions()

    for repetition, seed in enumerate(FORMAL_SEEDS, start=1):
        full_balanced_scenario = generator.generate(
            seed,
            FIXED_TASK_COMPOSITION,
            maximum_uav_count=MAXIMUM_UAV_COUNT,
        )
        for (
            experiment,
            condition,
            uav_count,
            battery_capacity,
            composition,
        ) in conditions:
            if experiment == EXPERIMENT_TASK_COUNT:
                scenario = nested_task_scenario(
                    full_balanced_scenario, int(condition)
                )
                actual_composition = {
                    task_type: sum(
                        task.task_type == task_type for task in scenario.tasks
                    )
                    for task_type in ("I", "M", "S", "N")
                }
            elif experiment == EXPERIMENT_COMPOSITION:
                scenario = generator.generate(
                    seed,
                    composition,
                    maximum_uav_count=MAXIMUM_UAV_COUNT,
                )
                actual_composition = dict(composition)
            else:
                scenario = full_balanced_scenario
                actual_composition = dict(FIXED_TASK_COMPOSITION)

            requested_task_count = sum(actual_composition.values())
            actual_task_count = len(scenario.tasks)
            if actual_task_count != requested_task_count:
                raise RuntimeError(
                    "The generated task count does not match the requested count"
                )

            scenario_hash = scenario_sha256(scenario)
            print(
                f"[{repetition}/{REPETITIONS}] {experiment}: {condition}; "
                f"tasks={actual_task_count}, UAVs={uav_count}, "
                f"battery={battery_capacity}",
                flush=True,
            )
            method_results = run_all_methods(
                scenario, uav_count, battery_capacity, seed
            )
            for method, details in method_results.items():
                completed_tasks = int(details["completed_tasks"])
                result_records.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "repetition": repetition,
                        "scenario_seed": seed,
                        "scenario_sha256": scenario_hash,
                        "grid_size": GRID_SIZE,
                        "requested_task_count": requested_task_count,
                        "actual_task_count": actual_task_count,
                        "I_task_count": actual_composition["I"],
                        "M_task_count": actual_composition["M"],
                        "S_task_count": actual_composition["S"],
                        "N_task_count": actual_composition["N"],
                        "uav_count": uav_count,
                        "initial_battery_capacity": battery_capacity,
                        "method": method,
                        "completed_tasks": completed_tasks,
                        "task_completion_rate": (
                            completed_tasks / actual_task_count
                        ),
                        "total_normalised_cost": details[
                            "total_normalised_cost"
                        ],
                        "total_remaining_energy": details[
                            "total_remaining_energy"
                        ],
                        "completed_tasks_by_uav": json.dumps(
                            details["completed_tasks_by_uav"]
                        ),
                    }
                )

            for task in scenario.tasks:
                scenario_records.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "repetition": repetition,
                        "scenario_seed": seed,
                        "scenario_sha256": scenario_hash,
                        "record_type": "task",
                        "identifier": task.task_id,
                        "grid_x": task.coordinate[0],
                        "grid_y": task.coordinate[1],
                        "task_type": task.task_type,
                        "original_longitude": _task_longitude(task),
                        "original_latitude": _task_latitude(task),
                        "source_record_id": task.source_record_id,
                    }
                )
            for uav_id, start in enumerate(
                scenario.uav_start_positions[:uav_count]
            ):
                scenario_records.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "repetition": repetition,
                        "scenario_seed": seed,
                        "scenario_sha256": scenario_hash,
                        "record_type": "uav_start",
                        "identifier": uav_id,
                        "grid_x": start[0],
                        "grid_y": start[1],
                        "task_type": None,
                        "original_longitude": None,
                        "original_latitude": None,
                        "source_record_id": None,
                    }
                )

    return pd.DataFrame(result_records), pd.DataFrame(scenario_records)


def summarise_results(raw_results):
    records = []
    metrics = ("completed_tasks", "task_completion_rate")
    for key, group in raw_results.groupby(
        ["experiment", "condition", "method"], sort=False
    ):
        experiment, condition, method = key
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            sample_size = len(values)
            mean = values.mean()
            standard_deviation = (
                values.std(ddof=1) if sample_size > 1 else 0.0
            )
            confidence_half_width = (
                t.ppf(0.975, sample_size - 1)
                * standard_deviation
                / math.sqrt(sample_size)
                if sample_size > 1
                else 0.0
            )
            records.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "method": method,
                    "metric": metric,
                    "sample_size": sample_size,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "confidence_interval_95_lower": mean
                    - confidence_half_width,
                    "confidence_interval_95_upper": mean
                    + confidence_half_width,
                }
            )
    return pd.DataFrame(records)


def holm_correction(p_values):
    count = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(count, dtype=float)
    current_maximum = 0.0
    for rank, original_position in enumerate(order):
        current_value = min(1.0, (count - rank) * p_values[original_position])
        current_maximum = max(current_maximum, current_value)
        corrected[original_position] = current_maximum
    return corrected.tolist()


def paired_significance_tests(raw_results):
    test_records = []
    grouped = raw_results.groupby(["experiment", "condition"], sort=False)
    for (experiment, condition), condition_table in grouped:
        metric = "completed_tasks"
        proposed = condition_table[
            condition_table["method"] == "Proposed method"
        ].set_index("scenario_seed")
        temporary_records = []
        for baseline_method in METHOD_ORDER[1:]:
            baseline = condition_table[
                condition_table["method"] == baseline_method
            ].set_index("scenario_seed")
            common_seeds = proposed.index.intersection(baseline.index)
            proposed_values = pd.to_numeric(proposed.loc[common_seeds, metric])
            baseline_values = pd.to_numeric(baseline.loc[common_seeds, metric])
            differences = proposed_values.to_numpy() - baseline_values.to_numpy()
            if len(differences) != REPETITIONS:
                raise RuntimeError(
                    "A paired test does not contain all ten common scenarios"
                )
            raw_p_value = (
                1.0
                if np.allclose(differences, 0.0)
                else float(
                    wilcoxon(differences, alternative="two-sided").pvalue
                )
            )
            temporary_records.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "metric": metric,
                    "proposed_method": "Proposed method",
                    "baseline_method": baseline_method,
                    "paired_sample_size": len(differences),
                    "proposed_method_mean": float(proposed_values.mean()),
                    "baseline_method_mean": float(baseline_values.mean()),
                    "mean_difference_proposed_minus_baseline": float(
                        np.mean(differences)
                    ),
                    "repetitions_where_proposed_is_better": int(
                        np.sum(differences > 0)
                    ),
                    "wilcoxon_raw_p_value": raw_p_value,
                }
            )

        corrected_values = holm_correction(
            [record["wilcoxon_raw_p_value"] for record in temporary_records]
        )
        for record, corrected_value in zip(
            temporary_records, corrected_values
        ):
            record["holm_corrected_p_value"] = corrected_value
            record["significant_after_correction_0_05"] = corrected_value < 0.05
            test_records.append(record)
    return pd.DataFrame(test_records)


def configure_plotting():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def save_svg(figure, output_path):
    output_path = Path(output_path).with_suffix(".svg")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def ordered_conditions(experiment):
    if experiment == EXPERIMENT_UAV_COUNT:
        return [str(value) for value in UAV_COUNT_LEVELS]
    if experiment == EXPERIMENT_BATTERY:
        return [str(value) for value in BATTERY_CAPACITY_LEVELS]
    if experiment == EXPERIMENT_TASK_COUNT:
        return [str(value) for value in TOTAL_TASK_COUNT_LEVELS]
    return list(TASK_COMPOSITION_LEVELS)


def display_condition(condition):
    if condition in TASK_COMPOSITION_LEVELS:
        composition = TASK_COMPOSITION_LEVELS[condition]
        return (
            f"{condition}\nI={composition['I']}  M={composition['M']}\n"
            f"S={composition['S']}  N={composition['N']}"
        )
    return str(condition)


def numeric_mean(values):
    return pd.to_numeric(values, errors="coerce").dropna().mean()


def numeric_standard_deviation(values):
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    return numeric_values.std(ddof=1) if len(numeric_values) > 1 else 0.0


def collect_plot_series(raw_results, experiment, metric, percentage=False):
    conditions = ordered_conditions(experiment)
    multiplier = 100.0 if percentage else 1.0
    series = {}
    for method in METHOD_ORDER:
        means = []
        standard_deviations = []
        for condition in conditions:
            values = raw_results[
                (raw_results["experiment"] == experiment)
                & (raw_results["condition"].astype(str) == str(condition))
                & (raw_results["method"] == method)
            ][metric]
            means.append(float(numeric_mean(values)) * multiplier)
            standard_deviations.append(
                float(numeric_standard_deviation(values)) * multiplier
            )
        series[method] = {
            "mean": np.array(means),
            "standard_deviation": np.array(standard_deviations),
        }
    return conditions, series


def _draw_series(axis, series, indices, methods=METHOD_ORDER):
    for method in methods:
        axis.errorbar(
            indices,
            series[method]["mean"][indices],
            yerr=series[method]["standard_deviation"][indices],
            label=method,
            color=METHOD_COLOURS[method],
            marker=METHOD_MARKERS[method],
            linestyle="-",
            linewidth=2.4 if method == "Proposed method" else 1.8,
            markersize=8,
            markerfacecolor=(
                METHOD_COLOURS[method]
                if method == "Proposed method"
                else "white"
            ),
            capsize=4,
            zorder=5 if method == "Proposed method" else 3,
        )


def plot_line_chart(
    raw_results,
    experiment,
    metric,
    title,
    x_label,
    y_label,
    output_path,
):
    conditions, series = collect_plot_series(raw_results, experiment, metric)
    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    indices = np.arange(len(conditions))
    _draw_series(axis, series, indices)
    axis.set_xticks(indices, [display_condition(value) for value in conditions])
    axis.set_xlabel(x_label, fontsize=13)
    axis.set_ylabel(y_label, fontsize=13)
    axis.set_title(title, fontsize=16, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=10, ncol=2, loc="upper left")
    figure.tight_layout()
    save_svg(figure, output_path)
    return conditions, series


def plot_line_detail(
    conditions,
    series,
    title,
    x_label,
    y_label,
    output_path,
):
    detail_indices = np.array([3, 4])
    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    _draw_series(axis, series, detail_indices, methods=DETAIL_METHODS)
    axis.set_xticks(
        detail_indices,
        [display_condition(conditions[index]) for index in detail_indices],
    )
    lower_values = []
    upper_values = []
    for method in DETAIL_METHODS:
        means = series[method]["mean"][detail_indices]
        deviations = series[method]["standard_deviation"][detail_indices]
        lower_values.extend(means - deviations)
        upper_values.extend(means + deviations)
    lower_bound = min(lower_values)
    upper_bound = max(upper_values)
    margin = max((upper_bound - lower_bound) * 0.15, 0.5)
    axis.set_ylim(lower_bound - margin, upper_bound + margin)
    axis.set_xlabel(x_label, fontsize=13)
    axis.set_ylabel(y_label, fontsize=13)
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=10, loc="best")
    figure.tight_layout()
    save_svg(figure, output_path)


def plot_task_composition(raw_results, output_path):
    conditions = ordered_conditions(EXPERIMENT_COMPOSITION)
    x_positions = np.arange(len(conditions), dtype=float)
    bar_width = 0.19
    figure, axis = plt.subplots(figsize=(12.8, 7.0))
    y_limit_candidates = []

    for method_index, method in enumerate(METHOD_ORDER):
        means = []
        standard_deviations = []
        for condition in conditions:
            values = raw_results[
                (raw_results["experiment"] == EXPERIMENT_COMPOSITION)
                & (raw_results["condition"].astype(str) == str(condition))
                & (raw_results["method"] == method)
            ]["completed_tasks"]
            means.append(float(numeric_mean(values)))
            standard_deviations.append(float(numeric_standard_deviation(values)))
        y_limit_candidates.extend(np.array(means) + np.array(standard_deviations))
        offset = (method_index - (len(METHOD_ORDER) - 1) / 2) * bar_width
        bars = axis.bar(
            x_positions + offset,
            means,
            width=bar_width,
            yerr=standard_deviations,
            capsize=4,
            color=METHOD_COLOURS[method],
            edgecolor="#333333",
            linewidth=0.8,
            label=method,
            zorder=3,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in means],
            padding=3,
            fontsize=8.5,
            rotation=90,
        )

    axis.set_xticks(
        x_positions, [display_condition(value) for value in conditions]
    )
    axis.set_xlabel("Task composition and count of each task type", fontsize=13)
    axis.set_ylabel("Mean completed tasks (error bars: SD)", fontsize=13)
    axis.set_ylim(0, max(y_limit_candidates) * 1.12)
    axis.grid(axis="y", alpha=0.25, zorder=0)
    handles, labels = axis.get_legend_handles_labels()
    figure.suptitle(
        "Effect of task composition on completed tasks",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        fontsize=10,
        ncol=4,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    save_svg(figure, output_path)


def plot_task_count(raw_results, output_path, detail_output_path):
    conditions, completed_series = collect_plot_series(
        raw_results, EXPERIMENT_TASK_COUNT, "completed_tasks"
    )
    _, completion_rate_series = collect_plot_series(
        raw_results,
        EXPERIMENT_TASK_COUNT,
        "task_completion_rate",
        percentage=True,
    )
    indices = np.arange(len(conditions))

    figure, (count_axis, rate_axis) = plt.subplots(
        1, 2, figsize=(14.2, 6.2), sharex=True
    )
    _draw_series(count_axis, completed_series, indices)
    _draw_series(rate_axis, completion_rate_series, indices)
    for axis in (count_axis, rate_axis):
        axis.set_xticks(indices, [display_condition(value) for value in conditions])
        axis.set_xlabel("Number of tasks", fontsize=12)
        axis.grid(alpha=0.25)
    count_axis.set_title("Mean completed tasks", fontsize=13, fontweight="bold")
    rate_axis.set_title("Mean task completion rate", fontsize=13, fontweight="bold")
    count_axis.set_ylabel("Mean completed tasks (error bars: SD)", fontsize=12)
    rate_axis.set_ylabel(
        "Mean task completion rate (%, error bars: SD)", fontsize=12
    )
    rate_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    handles, labels = count_axis.get_legend_handles_labels()
    figure.suptitle(
        "Effect of task count on task completion",
        fontsize=16,
        fontweight="bold",
    )
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    save_svg(figure, output_path)

    detail_indices = np.array([3, 4])
    detail_figure, (detail_count_axis, detail_rate_axis) = plt.subplots(
        1, 2, figsize=(12.8, 5.5), sharex=True
    )
    _draw_series(
        detail_count_axis,
        completed_series,
        detail_indices,
        methods=DETAIL_METHODS,
    )
    _draw_series(
        detail_rate_axis,
        completion_rate_series,
        detail_indices,
        methods=DETAIL_METHODS,
    )
    for axis in (detail_count_axis, detail_rate_axis):
        axis.set_xticks(
            detail_indices,
            [display_condition(conditions[index]) for index in detail_indices],
        )
        axis.set_xlabel("Number of tasks", fontsize=12)
        axis.grid(alpha=0.25)
    detail_count_axis.set_title("Completed tasks", fontsize=13, fontweight="bold")
    detail_rate_axis.set_title("Task completion rate", fontsize=13, fontweight="bold")
    detail_count_axis.set_ylabel(
        "Mean completed tasks (error bars: SD)", fontsize=12
    )
    detail_rate_axis.set_ylabel(
        "Mean completion rate (%, error bars: SD)", fontsize=12
    )
    detail_rate_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    detail_handles, detail_labels = detail_count_axis.get_legend_handles_labels()
    detail_figure.suptitle(
        "Detailed comparison at higher task counts",
        fontsize=15,
        fontweight="bold",
    )
    detail_figure.legend(
        detail_handles,
        detail_labels,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    detail_figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    save_svg(detail_figure, detail_output_path)


def plot_all_figures(raw_results, output_directory):
    configure_plotting()
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    uav_conditions, uav_series = plot_line_chart(
        raw_results,
        EXPERIMENT_UAV_COUNT,
        "completed_tasks",
        "Effect of UAV count on completed tasks",
        "Number of UAVs",
        "Mean completed tasks (error bars: SD)",
        figure_directory / "uav_count.svg",
    )
    plot_line_detail(
        uav_conditions,
        uav_series,
        "Detailed comparison at higher UAV counts",
        "Number of UAVs",
        "Mean completed tasks (error bars: SD)",
        figure_directory / "uav_count_detail.svg",
    )

    battery_conditions, battery_series = plot_line_chart(
        raw_results,
        EXPERIMENT_BATTERY,
        "completed_tasks",
        "Effect of battery capacity on completed tasks",
        "Initial battery capacity per UAV",
        "Mean completed tasks (error bars: SD)",
        figure_directory / "battery_capacity.svg",
    )
    plot_line_detail(
        battery_conditions,
        battery_series,
        "Detailed comparison at higher battery capacities",
        "Initial battery capacity per UAV",
        "Mean completed tasks (error bars: SD)",
        figure_directory / "battery_capacity_detail.svg",
    )

    plot_task_composition(
        raw_results, figure_directory / "task_composition.svg"
    )
    plot_task_count(
        raw_results,
        figure_directory / "task_count_completion.svg",
        figure_directory / "task_count_completion_detail.svg",
    )


def run_batch_experiments(output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=False)
    raw_results, scenario_details = generate_all_records()
    summary_statistics = summarise_results(raw_results)
    significance_tests = paired_significance_tests(raw_results)

    raw_results.to_csv(output_directory / "raw_results.csv", index=False)
    scenario_details.to_csv(output_directory / "scenario_details.csv", index=False)
    summary_statistics.to_csv(
        output_directory / "summary_statistics.csv", index=False
    )
    significance_tests.to_csv(
        output_directory / "paired_significance_tests.csv", index=False
    )
    plot_all_figures(raw_results, output_directory)
    return raw_results, summary_statistics, significance_tests
