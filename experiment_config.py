"""Fixed configuration for the four formal paired experiment families."""

from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_FILENAME = "hotosm_tur_destroyed_buildings_polygons_csv.csv"
DATA_PATH = PROJECT_DIRECTORY / "data" / DATA_FILENAME
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "experiment_results"

GRID_SIZE = 64
REPETITIONS = 10
FORMAL_SEEDS = tuple(2026081500 + index for index in range(1, REPETITIONS + 1))

EXECUTION_LOADS = {"I": 6, "M": 5, "S": 3, "N": 2}

FIXED_TASK_COUNT = 200
FIXED_TASK_COMPOSITION = {"I": 50, "M": 50, "S": 50, "N": 50}
FIXED_UAV_COUNT = 10
FIXED_BATTERY_CAPACITY = 160
MAXIMUM_UAV_COUNT = 10

UAV_COUNT_LEVELS = (2, 4, 6, 8, 10)
BATTERY_CAPACITY_LEVELS = (80, 120, 160, 200, 240)
TOTAL_TASK_COUNT_LEVELS = (40, 80, 120, 160, 200)

TASK_COMPOSITION_LEVELS = {
    "Balanced": {"I": 50, "M": 50, "S": 50, "N": 50},
    "More I tasks": {"I": 100, "M": 40, "S": 30, "N": 30},
    "More M tasks": {"I": 30, "M": 100, "S": 40, "N": 30},
    "More low load tasks": {"I": 30, "M": 30, "S": 70, "N": 70},
}

METHOD_ORDER = (
    "Proposed method",
    "Original consensus auction",
    "Nearest feasible task first",
    "Random feasible allocation",
)
