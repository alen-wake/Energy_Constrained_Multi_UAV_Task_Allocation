"""Original single-task consensus auction baseline.

This module implements the core single-task process described by Choi,
Brunet, and How (2009): local bidding, exchange of local winner and bid
records, maximum consensus, and rebidding by UAVs that lose an auction.
Every UAV maintains independent local tables. The experiments assume
synchronous, fully connected, and reliable communication.
"""

from baseline import calculate_task_cost, run_round_based_baseline
from model import LocalAuctionState


COMPARISON_TOLERANCE = 1e-12
DEFAULT_GRID_SIZE = 20


def calculate_fixed_task_reward(execution_loads, grid_size=DEFAULT_GRID_SIZE):
    """Give every feasible task a positive cost-based reward within the grid."""
    maximum_manhattan_distance = 2 * (grid_size - 1)
    return float(maximum_manhattan_distance + max(execution_loads.values()) + 1)


def _record_priority(winning_bid, winner_id):
    """Prefer the higher bid and then the lower UAV identifier."""
    identifier_order = -winner_id if winner_id is not None else float("-inf")
    return winning_bid, identifier_order


def _exchange_and_merge_local_tables(local_states, task_ids):
    """Simulate one synchronous, fully connected, reliable consensus exchange."""
    snapshots = [
        (state.bid_table.copy(), state.winner_table.copy())
        for state in local_states
    ]

    merged_bids = {}
    merged_winners = {}
    for task_id in task_ids:
        best_bid, best_winner = max(
            (
                (bid_table[task_id], winner_table[task_id])
                for bid_table, winner_table in snapshots
            ),
            key=lambda record: _record_priority(*record),
        )
        merged_bids[task_id] = best_bid
        merged_winners[task_id] = best_winner

    for state in local_states:
        state.bid_table = merged_bids.copy()
        state.winner_table = merged_winners.copy()


def _local_tables_agree(local_states, task_ids):
    if not local_states:
        return True
    reference_state = local_states[0]
    return all(
        state.winner_table[task_id] == reference_state.winner_table[task_id]
        and abs(state.bid_table[task_id] - reference_state.bid_table[task_id])
        <= COMPARISON_TOLERANCE
        for state in local_states[1:]
        for task_id in task_ids
    )


def generate_original_consensus_auction_round_assignment(
    pending_tasks,
    uavs,
    execution_loads,
    return_details=False,
    grid_size=DEFAULT_GRID_SIZE,
):
    """Complete local bidding and maximum consensus for one execution round."""
    task_ids = [task.task_id for task in pending_tasks]
    local_states = [
        LocalAuctionState(
            uav_id=uav_id,
            bid_table={task_id: 0.0 for task_id in task_ids},
            winner_table={task_id: None for task_id in task_ids},
        )
        for uav_id in range(len(uavs))
    ]
    fixed_reward = calculate_fixed_task_reward(execution_loads, grid_size)
    auction_iterations = 0
    maximum_iterations = max(1, len(uavs) * max(1, len(pending_tasks)) + 1)

    for _ in range(maximum_iterations):
        new_bid_in_round = False

        for uav_id, uav in enumerate(uavs):
            local_state = local_states[uav_id]
            already_has_task = any(
                winner_id == uav_id
                for winner_id in local_state.winner_table.values()
            )
            if already_has_task:
                continue

            candidate_tasks = []
            for task in pending_tasks:
                cost = calculate_task_cost(uav, task, execution_loads)
                if cost > uav.remaining_energy:
                    continue
                task_reward = fixed_reward - cost
                current_bid = local_state.bid_table[task.task_id]
                if task_reward > current_bid + COMPARISON_TOLERANCE:
                    candidate_tasks.append((task_reward, task, cost))

            if not candidate_tasks:
                continue

            task_reward, task, _ = max(
                candidate_tasks,
                key=lambda candidate: (candidate[0], -candidate[1].task_id),
            )
            local_state.bid_table[task.task_id] = task_reward
            local_state.winner_table[task.task_id] = uav_id
            new_bid_in_round = True

        if not new_bid_in_round:
            break

        auction_iterations += 1
        _exchange_and_merge_local_tables(local_states, task_ids)
        if not _local_tables_agree(local_states, task_ids):
            raise RuntimeError(
                "Local tables disagree after a reliable consensus exchange"
            )
    else:
        raise RuntimeError(
            "The original consensus auction exceeded its iteration limit"
        )

    final_state = local_states[0] if local_states else None
    assignments = []
    if final_state is not None:
        for task in pending_tasks:
            uav_id = final_state.winner_table[task.task_id]
            if uav_id is not None:
                cost = calculate_task_cost(uavs[uav_id], task, execution_loads)
                assignments.append((uav_id, task, cost))

    if not return_details:
        return assignments
    return assignments, {
        "auction_iterations": auction_iterations,
        "local_winner_tables": [
            state.winner_table.copy() for state in local_states
        ],
        "local_bid_tables": [state.bid_table.copy() for state in local_states],
        "local_tables_agree": _local_tables_agree(local_states, task_ids),
    }


def run_original_consensus_auction(
    scenario,
    uav_count,
    battery_capacity,
    execution_loads,
    return_details=False,
):
    """Run the original single-task consensus auction round by round."""
    maximum_coordinate = max(
        [value for task in scenario.tasks for value in task.coordinate]
        + [
            value
            for start in scenario.uav_start_positions[:uav_count]
            for value in start
        ]
        + [DEFAULT_GRID_SIZE - 1]
    )
    scenario_grid_size = maximum_coordinate + 1

    def allocate_one_round(pending_tasks, uavs, current_execution_loads):
        return generate_original_consensus_auction_round_assignment(
            pending_tasks,
            uavs,
            current_execution_loads,
            grid_size=scenario_grid_size,
        )

    return run_round_based_baseline(
        scenario,
        uav_count,
        battery_capacity,
        execution_loads,
        allocate_one_round,
        return_details,
    )
