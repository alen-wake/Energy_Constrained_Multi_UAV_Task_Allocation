"""Cost calculation and round-based execution shared by the three baselines."""

from model import UAVState


def calculate_task_cost(uav, task, execution_loads):
    """Return Manhattan travel cost plus the task execution load."""
    travel_distance = abs(uav.coordinate[0] - task.coordinate[0]) + abs(
        uav.coordinate[1] - task.coordinate[1]
    )
    return float(travel_distance + execution_loads[task.task_type])


def create_uav_states(scenario, uav_count, battery_capacity):
    """Create UAV states from a scenario without modifying the scenario."""
    if uav_count > len(scenario.uav_start_positions):
        raise ValueError("The scenario has fewer UAV start positions than requested")
    return [
        UAVState(
            uav_id,
            scenario.uav_start_positions[uav_id],
            float(battery_capacity),
        )
        for uav_id in range(uav_count)
    ]


def validate_and_execute_round(assignments, pending_tasks, uavs, execution_loads):
    """Validate one-to-one and energy constraints, then update system state."""
    uav_ids = [uav_id for uav_id, _, _ in assignments]
    task_ids = [task.task_id for _, task, _ in assignments]
    if len(uav_ids) != len(set(uav_ids)):
        raise RuntimeError("A UAV received more than one task in one execution round")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("A task was assigned to more than one UAV in one round")

    for uav_id, task, reported_cost in assignments:
        if task not in pending_tasks:
            raise RuntimeError("An assignment contains a task that is not pending")
        uav = uavs[uav_id]
        actual_cost = calculate_task_cost(uav, task, execution_loads)
        if abs(actual_cost - reported_cost) > 1e-9:
            raise RuntimeError(
                "An assignment cost does not match the common cost function"
            )
        if actual_cost > uav.remaining_energy + 1e-9:
            raise RuntimeError("An assignment violates the UAV energy constraint")

    for uav_id, task, cost in assignments:
        uav = uavs[uav_id]
        uav.remaining_energy -= cost
        uav.coordinate = task.coordinate
        pending_tasks.remove(task)


def run_round_based_baseline(
    scenario,
    uav_count,
    battery_capacity,
    execution_loads,
    allocate_one_round,
    return_details=False,
):
    """Repeat one-to-one allocation rounds and return common result fields."""
    pending_tasks = list(scenario.tasks)
    uavs = create_uav_states(scenario, uav_count, battery_capacity)
    completed_tasks = 0
    total_cost = 0.0
    completed_by_uav = [0 for _ in uavs]

    while pending_tasks:
        assignments = allocate_one_round(pending_tasks, uavs, execution_loads)
        if not assignments:
            break
        validate_and_execute_round(assignments, pending_tasks, uavs, execution_loads)
        completed_tasks += len(assignments)
        total_cost += sum(cost for _, _, cost in assignments)
        for uav_id, _, _ in assignments:
            completed_by_uav[uav_id] += 1

    if not return_details:
        return completed_tasks
    return {
        "completed_tasks": completed_tasks,
        "total_normalised_cost": total_cost,
        "completed_tasks_by_uav": tuple(completed_by_uav),
        "total_remaining_energy": sum(uav.remaining_energy for uav in uavs),
    }
