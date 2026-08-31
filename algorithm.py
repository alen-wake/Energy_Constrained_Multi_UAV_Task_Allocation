"""Energy-constrained consensus bundle allocation.

The communication and bundle-update process follows the CBBA framework of
Choi, Brunet, and How and the public MIT ACL reference implementation. Each
UAV independently maintains a bundle, route, winner table, bid table, and
communication timestamps. This project adapts candidate scoring and energy
feasibility to its task model. The implementation is not a line-by-line
translation of the reference code.
"""

from dataclasses import dataclass, field


TOLERANCE = 1e-9


def manhattan_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def route_cost(start, route, execution_loads):
    current_position = start
    total_cost = 0.0
    for task in route:
        total_cost += (
            manhattan_distance(current_position, task.coordinate)
            + execution_loads[task.task_type]
        )
        current_position = task.coordinate
    return float(total_cost)


def best_insertion(start, current_route, candidate_task, execution_loads):
    original_cost = route_cost(start, current_route, execution_loads)
    best_record = None
    for position in range(len(current_route) + 1):
        new_route = list(current_route)
        new_route.insert(position, candidate_task)
        new_cost = route_cost(start, new_route, execution_loads)
        record = (
            new_cost - original_cost,
            position,
            tuple(new_route),
            new_cost,
        )
        if best_record is None or record[:2] < best_record[:2]:
            best_record = record
    return best_record


def _is_better_record(new_bid, new_winner, old_bid, old_winner):
    if new_bid > old_bid + TOLERANCE:
        return True
    return (
        abs(new_bid - old_bid) <= TOLERANCE
        and new_winner is not None
        and (old_winner is None or new_winner < old_winner)
    )


@dataclass
class LocalBundleState:
    uav_id: int
    start: tuple[int, int]
    energy_budget: float
    bundle: list[int] = field(default_factory=list)
    route: list = field(default_factory=list)
    route_scores: list[float] = field(default_factory=list)
    winner_table: dict[int, int | None] = field(default_factory=dict)
    bid_table: dict[int, float] = field(default_factory=dict)


def _reliable_fully_connected_communication(
    states, timestamps, current_round, task_ids
):
    """Exchange local records on a synchronous fully connected graph."""
    uav_count = len(states)
    old_winners = [state.winner_table.copy() for state in states]
    old_bids = [state.bid_table.copy() for state in states]
    old_timestamps = [row.copy() for row in timestamps]
    new_winners = [state.winner_table.copy() for state in states]
    new_bids = [state.bid_table.copy() for state in states]
    new_timestamps = [row.copy() for row in timestamps]

    def update(receiver, task_id, sender_winner, sender_bid):
        new_winners[receiver][task_id] = sender_winner
        new_bids[receiver][task_id] = sender_bid

    def reset(receiver, task_id):
        new_winners[receiver][task_id] = None
        new_bids[receiver][task_id] = 0.0

    for sender in range(uav_count):
        for receiver in range(uav_count):
            if sender == receiver:
                continue
            for task_id in task_ids:
                sender_winner = old_winners[sender][task_id]
                receiver_winner = new_winners[receiver][task_id]
                sender_bid = old_bids[sender][task_id]
                receiver_bid = new_bids[receiver][task_id]

                if sender_winner == sender:
                    if receiver_winner == receiver:
                        if _is_better_record(
                            sender_bid, sender, receiver_bid, receiver
                        ):
                            update(receiver, task_id, sender, sender_bid)
                    elif receiver_winner == sender:
                        update(receiver, task_id, sender, sender_bid)
                    elif receiver_winner is not None:
                        if (
                            old_timestamps[sender][receiver_winner]
                            > new_timestamps[receiver][receiver_winner]
                            or _is_better_record(
                                sender_bid,
                                sender,
                                receiver_bid,
                                receiver_winner,
                            )
                        ):
                            update(receiver, task_id, sender, sender_bid)
                    else:
                        update(receiver, task_id, sender, sender_bid)

                elif sender_winner == receiver:
                    if receiver_winner == sender:
                        reset(receiver, task_id)
                    elif receiver_winner is not None and receiver_winner != receiver:
                        if (
                            old_timestamps[sender][receiver_winner]
                            > new_timestamps[receiver][receiver_winner]
                        ):
                            reset(receiver, task_id)

                elif sender_winner is not None:
                    third_party = sender_winner
                    if receiver_winner == receiver:
                        if (
                            old_timestamps[sender][third_party]
                            > new_timestamps[receiver][third_party]
                            and _is_better_record(
                                sender_bid,
                                third_party,
                                receiver_bid,
                                receiver,
                            )
                        ):
                            update(receiver, task_id, third_party, sender_bid)
                    elif receiver_winner == sender:
                        if (
                            old_timestamps[sender][third_party]
                            > new_timestamps[receiver][third_party]
                        ):
                            update(receiver, task_id, third_party, sender_bid)
                        else:
                            reset(receiver, task_id)
                    elif receiver_winner == third_party:
                        if (
                            old_timestamps[sender][third_party]
                            > new_timestamps[receiver][third_party]
                        ):
                            update(receiver, task_id, third_party, sender_bid)
                    elif receiver_winner is not None:
                        old_third_party = receiver_winner
                        if (
                            old_timestamps[sender][old_third_party]
                            > new_timestamps[receiver][old_third_party]
                        ):
                            if (
                                old_timestamps[sender][third_party]
                                >= new_timestamps[receiver][third_party]
                            ):
                                update(receiver, task_id, third_party, sender_bid)
                            else:
                                reset(receiver, task_id)
                        elif (
                            old_timestamps[sender][third_party]
                            > new_timestamps[receiver][third_party]
                            and _is_better_record(
                                sender_bid,
                                third_party,
                                receiver_bid,
                                old_third_party,
                            )
                        ):
                            update(receiver, task_id, third_party, sender_bid)
                    elif (
                        old_timestamps[sender][third_party]
                        > new_timestamps[receiver][third_party]
                    ):
                        update(receiver, task_id, third_party, sender_bid)

                else:
                    if receiver_winner == sender:
                        reset(receiver, task_id)
                    elif receiver_winner is not None and receiver_winner != receiver:
                        if (
                            old_timestamps[sender][receiver_winner]
                            > new_timestamps[receiver][receiver_winner]
                        ):
                            reset(receiver, task_id)

            for other_uav in range(uav_count):
                if (
                    other_uav != receiver
                    and new_timestamps[receiver][other_uav]
                    < old_timestamps[sender][other_uav]
                ):
                    new_timestamps[receiver][other_uav] = old_timestamps[sender][
                        other_uav
                    ]
            new_timestamps[receiver][sender] = current_round

    for uav_id, state in enumerate(states):
        state.winner_table = new_winners[uav_id]
        state.bid_table = new_bids[uav_id]
    return new_timestamps


def _release_lost_task_suffix(state):
    first_lost_position = None
    for position, task_id in enumerate(state.bundle):
        if state.winner_table[task_id] != state.uav_id:
            first_lost_position = position
            break
    if first_lost_position is None:
        return False

    released_ids = set(state.bundle[first_lost_position:])
    for task_id in released_ids:
        if state.winner_table[task_id] == state.uav_id:
            state.winner_table[task_id] = None
            state.bid_table[task_id] = 0.0

    retained_positions = [
        index
        for index, task in enumerate(state.route)
        if task.task_id not in released_ids
    ]
    state.route = [state.route[index] for index in retained_positions]
    state.route_scores = [
        state.route_scores[index] for index in retained_positions
    ]
    state.bundle = state.bundle[:first_lost_position]
    return True


def _extend_bundle(state, all_tasks, execution_loads, fixed_reward):
    made_new_bid = False
    added_tasks = set(state.bundle)
    while True:
        previous_bundle_bid = (
            state.bid_table[state.bundle[-1]] if state.bundle else float("inf")
        )
        candidates = []
        for task in all_tasks:
            if task.task_id in added_tasks:
                continue
            incremental_cost, insertion_position, new_route, new_total_cost = (
                best_insertion(state.start, state.route, task, execution_loads)
            )
            if new_total_cost > state.energy_budget + TOLERANCE:
                continue
            marginal_score = min(
                fixed_reward - incremental_cost, previous_bundle_bid
            )
            current_winner = state.winner_table[task.task_id]
            current_bid = state.bid_table[task.task_id]
            if not (
                marginal_score > current_bid + TOLERANCE
                or (
                    abs(marginal_score - current_bid) <= TOLERANCE
                    and (current_winner is None or state.uav_id < current_winner)
                )
            ):
                continue
            candidates.append(
                (
                    marginal_score,
                    -incremental_cost,
                    -task.coordinate[0],
                    -task.coordinate[1],
                    task,
                    insertion_position,
                    new_route,
                )
            )

        if not candidates:
            break
        marginal_score, _, _, _, task, insertion_position, new_route = max(
            candidates, key=lambda item: item[:4]
        )
        state.bundle.append(task.task_id)
        state.route = list(new_route)
        state.route_scores.insert(insertion_position, float(marginal_score))
        state.winner_table[task.task_id] = state.uav_id
        state.bid_table[task.task_id] = float(marginal_score)
        added_tasks.add(task.task_id)
        made_new_bid = True
    return made_new_bid


def _calculate_fixed_reward(tasks, uav_start_positions, execution_loads):
    maximum_coordinate = max(
        [value for task in tasks for value in task.coordinate]
        + [value for start in uav_start_positions for value in start]
        + [1]
    )
    return float(4 * maximum_coordinate + max(execution_loads.values()) + 1)


def generate_local_bundle_preview(
    scenario, uav_count, battery_capacity, execution_loads
):
    """Build each UAV's local bundle before winner records are exchanged."""
    if uav_count > len(scenario.uav_start_positions):
        raise ValueError("The scenario has fewer UAV start positions than requested")
    task_ids = [task.task_id for task in scenario.tasks]
    fixed_reward = _calculate_fixed_reward(
        scenario.tasks, scenario.uav_start_positions[:uav_count], execution_loads
    )
    states = [
        LocalBundleState(
            uav_id=uav_id,
            start=scenario.uav_start_positions[uav_id],
            energy_budget=float(battery_capacity),
            winner_table={task_id: None for task_id in task_ids},
            bid_table={task_id: 0.0 for task_id in task_ids},
        )
        for uav_id in range(uav_count)
    ]
    for state in states:
        _extend_bundle(state, scenario.tasks, execution_loads, fixed_reward)
    return {
        "fixed_reward": fixed_reward,
        "bundles_by_uav": tuple(tuple(state.bundle) for state in states),
        "routes_by_uav": tuple(
            tuple(task.task_id for task in state.route) for state in states
        ),
        "route_costs_by_uav": tuple(
            route_cost(state.start, state.route, execution_loads)
            for state in states
        ),
    }


def run_energy_constrained_bundle(
    scenario,
    uav_count,
    battery_capacity,
    execution_loads,
    return_details=False,
):
    if uav_count > len(scenario.uav_start_positions):
        raise ValueError("The scenario has fewer UAV start positions than requested")
    task_ids = [task.task_id for task in scenario.tasks]
    states = [
        LocalBundleState(
            uav_id=uav_id,
            start=scenario.uav_start_positions[uav_id],
            energy_budget=float(battery_capacity),
            winner_table={task_id: None for task_id in task_ids},
            bid_table={task_id: 0.0 for task_id in task_ids},
        )
        for uav_id in range(uav_count)
    ]
    fixed_reward = _calculate_fixed_reward(
        scenario.tasks, scenario.uav_start_positions[:uav_count], execution_loads
    )
    timestamps = [[0 for _ in range(uav_count)] for _ in range(uav_count)]
    current_round = 1
    last_round_with_new_bid = 0
    broadcast_messages = 0
    maximum_rounds = max(50, 5 * uav_count * max(1, len(task_ids)))

    while current_round <= maximum_rounds:
        timestamps = _reliable_fully_connected_communication(
            states, timestamps, current_round, task_ids
        )
        broadcast_messages += uav_count * max(0, uav_count - 1)
        new_bid_in_round = False
        for state in states:
            _release_lost_task_suffix(state)
            new_bid_in_round = (
                _extend_bundle(state, scenario.tasks, execution_loads, fixed_reward)
                or new_bid_in_round
            )
        if new_bid_in_round:
            last_round_with_new_bid = current_round
        if current_round - last_round_with_new_bid > uav_count:
            break
        current_round += 1
    else:
        raise RuntimeError("The bundle auction did not converge within its limit")

    timestamps = _reliable_fully_connected_communication(
        states, timestamps, current_round + 1, task_ids
    )
    broadcast_messages += uav_count * max(0, uav_count - 1)
    for state in states:
        _release_lost_task_suffix(state)

    reference_winners = states[0].winner_table if states else {}
    reference_bids = states[0].bid_table if states else {}
    if any(
        state.winner_table != reference_winners
        or state.bid_table != reference_bids
        for state in states[1:]
    ):
        raise RuntimeError(
            "Local winner or bid tables disagree after reliable communication"
        )

    assigned_task_ids = [
        task.task_id for state in states for task in state.route
    ]
    if len(assigned_task_ids) != len(set(assigned_task_ids)):
        raise RuntimeError("The bundle result contains duplicate task assignments")
    costs_by_uav = [
        route_cost(state.start, state.route, execution_loads) for state in states
    ]
    if any(cost > battery_capacity + TOLERANCE for cost in costs_by_uav):
        raise RuntimeError("The bundle result violates the UAV energy constraint")

    completed_tasks = len(assigned_task_ids)
    total_cost = sum(costs_by_uav)
    if not return_details:
        return completed_tasks
    return {
        "completed_tasks": completed_tasks,
        "total_normalised_cost": total_cost,
        "completed_tasks_by_uav": tuple(len(state.route) for state in states),
        "costs_by_uav": tuple(costs_by_uav),
        "bundles_by_uav": tuple(tuple(state.bundle) for state in states),
        "routes_by_uav": tuple(
            tuple(task.task_id for task in state.route) for state in states
        ),
        "route_scores_by_uav": tuple(
            tuple(state.route_scores) for state in states
        ),
        "fixed_reward": fixed_reward,
        "total_remaining_energy": sum(
            battery_capacity - cost for cost in costs_by_uav
        ),
        "consensus_rounds": current_round + 1,
        "broadcast_messages": broadcast_messages,
    }
