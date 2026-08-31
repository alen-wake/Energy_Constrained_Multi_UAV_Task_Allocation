# Energy Constrained Multi UAV Task Allocation

This repository contains the software developed for an MSc dissertation on
energy constrained task allocation for multiple unmanned aerial vehicles
(UAVs). It provides the proposed allocation method, three baseline methods,
four paired experiment families, and an interactive demonstration. The batch
experiment and the demonstration use the same implementation of the proposed
method.

## Methods

The experiment compares four methods:

1. the proposed Energy Constrained Consensus Based Bundle Method;
2. the Original Consensus Auction baseline;
3. nearest feasible task first;
4. random feasible allocation.

The proposed method builds ordered task bundles, applies route insertion,
checks the energy cost of the complete route, exchanges independent local
winner and bid records, and releases a bundle suffix after losing a task.

## Repository contents

- `algorithm.py`: proposed energy-constrained consensus bundle method;
- `original_consensus_auction_baseline.py`: original single-task consensus
  auction baseline;
- `nearest_feasible_task_baseline.py`: nearest feasible task first baseline;
- `random_feasible_allocation_baseline.py`: random feasible allocation baseline;
- `baseline.py`: common cost and round-execution functions for the baselines;
- `model.py`: task, scenario, UAV, and local auction data structures;
- `data_scenario.py`: read-only data loading and grid scenario generation;
- `experiment_config.py`: fixed formal experiment settings;
- `batch_experiments.py`: paired experiments, statistics, and SVG figures;
- `run_batch.py`: command-line entry point for the formal experiment;
- `demo.py`: interactive three-step explanation of the proposed method.

The repository does not include generated experiment results, cache files, or
the source dataset.

## Data requirement

The program uses the **HOTOSM Turkey Destroyed Buildings (OpenStreetMap
Export)** dataset published through the Humanitarian Data Exchange:

https://data.humdata.org/dataset/41765491-7345-421f-91d8-a023412e46b5

Download the CSV resource and save it at:

```text
data/hotosm_tur_destroyed_buildings_polygons_csv.csv
```

The program reads `osm_id`, `longitude`, and `latitude`. It does not modify the
source file. Generated scenarios and experiment results are written to a
separate output directory. OpenStreetMap attribution and licence information
are available at https://www.openstreetmap.org/copyright.

## Formal experiment design

The four experiment families vary:

1. UAV count: 2, 4, 6, 8, and 10;
2. initial battery capacity per UAV: 80, 120, 160, 200, and 240;
3. total task count: 40, 80, 120, 160, and 200;
4. task composition: balanced, more I tasks, more M tasks, and more low-load
   tasks.

There are 19 conditions. Unless one variable is changed, the standard setting
uses a 64 by 64 grid, 200 tasks, 10 UAVs, and an initial battery capacity of
160 per UAV. Each condition has 10 paired repetitions. All four methods receive
identical task locations, task types, and UAV starting positions within one
repetition. A complete run therefore produces 190 paired scenarios and 760
method runs.

## Installation

Python 3.13 was used for the dissertation experiments. Install the required
packages from the repository directory:

```powershell
python -m pip install -r requirements.txt
```

## Run the formal experiments

From this directory, run:

```powershell
python run_batch.py
```

To choose the output directory:

```powershell
python run_batch.py --output "D:\path\to\output"
```

Each output directory contains:

- `raw_results.csv`;
- `scenario_details.csv`;
- `summary_statistics.csv`;
- `paired_significance_tests.csv`;
- a `figures` directory containing seven editable SVG figures.

The four main figures contain no embedded inset. Three additional detail
figures show the closely grouped non-random methods at the final two settings
of the UAV-count, battery-capacity, and task-count experiments.

## Run the demonstration

```powershell
python demo.py
```

The demonstration shows local bundle construction, exchange and conflict
resolution, and execution of the agreed paths. It imports the same
`algorithm.py` implementation as the formal batch experiment. The interface is
an explanatory tool and does not provide formal experimental data.

## Reproducibility and scope

Every condition uses 10 recorded scenario seeds. Within one repetition, all
four methods receive identical task locations, task types, and UAV starting
positions. The software checks that final routes satisfy their energy budgets
and that no task appears in more than one final route.

The model uses a 64 by 64 grid, Manhattan travel cost, relative task execution
loads, homogeneous UAVs, and synchronous, fully connected, reliable
communication. It does not simulate flight dynamics, obstacles, packet loss,
delays, charging, or dynamically arriving tasks.
