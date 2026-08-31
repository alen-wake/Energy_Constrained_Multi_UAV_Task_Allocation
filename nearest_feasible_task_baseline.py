"""Nearest feasible task first baseline.

The local choice follows the classic nearest-neighbour heuristic, adapted to
multiple UAVs, round-based execution, and an energy-feasibility constraint.
It is not a complete reproduction of the travelling salesperson algorithm.
"""

from baseline import calculate_task_cost, run_round_based_baseline


def _travel_distance(uav, task):
    return abs(uav.coordinate[0] - task.coordinate[0]) + abs(
        uav.coordinate[1] - task.coordinate[1]
    )


def generate_nearest_task_round_assignment(pending_tasks, uavs, execution_loads):
    """Assign each UAV its nearest currently feasible unallocated task."""
    available_tasks = list(pending_tasks)
    assignments = []
    uav_order = sorted(
        range(len(uavs)),
        key=lambda uav_id: (
            uavs[uav_id].coordinate[0],
            uavs[uav_id].coordinate[1],
            uav_id,
        ),
    )

    for uav_id in uav_order:
        uav = uavs[uav_id]
        candidates = []
        for task in available_tasks:
            cost = calculate_task_cost(uav, task, execution_loads)
            if cost <= uav.remaining_energy:
                candidates.append((task, cost, _travel_distance(uav, task)))
        if not candidates:
            continue

        task, cost, _ = min(
            candidates,
            key=lambda item: (
                item[2],
                item[0].coordinate[0],
                item[0].coordinate[1],
                item[0].task_type,
            ),
        )
        assignments.append((uav_id, task, cost))
        available_tasks.remove(task)

    return assignments


def run_nearest_feasible_task_first(
    scenario,
    uav_count,
    battery_capacity,
    execution_loads,
    return_details=False,
):
    """Run nearest feasible task first until no feasible task remains."""
    return run_round_based_baseline(
        scenario,
        uav_count,
        battery_capacity,
        execution_loads,
        generate_nearest_task_round_assignment,
        return_details,
    )
