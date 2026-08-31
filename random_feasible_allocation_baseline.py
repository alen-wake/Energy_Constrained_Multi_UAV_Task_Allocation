"""Random feasible allocation baseline defined for this study."""

import random

from baseline import calculate_task_cost, run_round_based_baseline


def generate_random_round_assignment(
    pending_tasks, uavs, execution_loads, random_generator
):
    """Generate a conflict-free random feasible matching for one round."""
    available_uavs = set(range(len(uavs)))
    available_tasks = list(pending_tasks)
    assignments = []

    while available_uavs and available_tasks:
        feasible_pairs = []
        for uav_id in sorted(available_uavs):
            uav = uavs[uav_id]
            for task in available_tasks:
                cost = calculate_task_cost(uav, task, execution_loads)
                if cost <= uav.remaining_energy:
                    feasible_pairs.append((uav_id, task, cost))

        if not feasible_pairs:
            break

        selected_pair = random_generator.choice(feasible_pairs)
        assignments.append(selected_pair)
        available_uavs.remove(selected_pair[0])
        available_tasks.remove(selected_pair[1])

    return assignments


def run_random_feasible_allocation(
    scenario,
    uav_count,
    battery_capacity,
    execution_loads,
    random_seed,
    return_details=False,
):
    """Run all execution rounds with a fixed random seed."""
    random_generator = random.Random(random_seed)

    def allocate_one_round(pending_tasks, uavs, current_execution_loads):
        return generate_random_round_assignment(
            pending_tasks, uavs, current_execution_loads, random_generator
        )

    return run_round_based_baseline(
        scenario,
        uav_count,
        battery_capacity,
        execution_loads,
        allocate_one_round,
        return_details,
    )
