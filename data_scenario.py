"""Read real data without modification and generate unique grid task locations."""

import random
from pathlib import Path

import pandas as pd

from model import Scenario, Task, TASK_TYPES


class RealDataScenarioGenerator:
    def __init__(self, csv_path, grid_size=20):
        data = pd.read_csv(
            Path(csv_path),
            usecols=["osm_id", "longitude", "latitude"],
            dtype={"osm_id": "string"},
        ).dropna(subset=["longitude", "latitude"]).drop_duplicates(
            subset=["longitude", "latitude"]
        )
        self.data = data.reset_index(drop=True)
        self.grid_size = grid_size

    def _window_records(self, longitude, latitude, span):
        window = self.data[
            self.data.longitude.between(longitude - span / 2, longitude + span / 2)
            & self.data.latitude.between(latitude - span / 2, latitude + span / 2)
        ]
        grid_x = (
            (window.longitude - (longitude - span / 2)) / span * self.grid_size
        ).astype(int).clip(0, self.grid_size - 1)
        grid_y = (
            (window.latitude - (latitude - span / 2)) / span * self.grid_size
        ).astype(int).clip(0, self.grid_size - 1)

        records = {}
        for (_, row), x_value, y_value in zip(
            window.iterrows(), grid_x.tolist(), grid_y.tolist()
        ):
            records.setdefault(
                (x_value, y_value),
                {
                    "original_coordinate": (
                        float(row.longitude),
                        float(row.latitude),
                    ),
                    "source_record_id": str(row.osm_id),
                },
            )
        return records

    def generate(self, seed, task_type_counts, maximum_uav_count):
        total_tasks = sum(task_type_counts.values())
        random_generator = random.Random(seed)
        if total_tasks > self.grid_size**2:
            raise ValueError("The requested task count exceeds the number of grid cells")

        coordinates = self.data[["longitude", "latitude"]].to_numpy()
        for span_factor in (1, 2, 4, 8, 16, 32):
            for _ in range(150):
                longitude, latitude = coordinates[
                    random_generator.randrange(len(coordinates))
                ]
                cell_records = self._window_records(
                    longitude, latitude, 0.02 * span_factor
                )
                if len(cell_records) >= total_tasks:
                    selected_cells = random_generator.sample(
                        list(cell_records), total_tasks
                    )
                    random_generator.shuffle(selected_cells)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(
                "The real data do not contain enough unique grid locations; "
                "the task set was not generated or truncated"
            )

        tasks = []
        task_index = 0
        for task_type in TASK_TYPES:
            for _ in range(task_type_counts[task_type]):
                source = cell_records[selected_cells[task_index]]
                tasks.append(
                    Task(
                        task_index,
                        selected_cells[task_index],
                        task_type,
                        source["original_coordinate"],
                        source["source_record_id"],
                    )
                )
                task_index += 1

        occupied_cells = set(selected_cells)
        free_cells = [
            (x_value, y_value)
            for x_value in range(self.grid_size)
            for y_value in range(self.grid_size)
            if (x_value, y_value) not in occupied_cells
        ]
        start_positions = tuple(
            random_generator.sample(free_cells, maximum_uav_count)
        )
        return Scenario(seed, tuple(tasks), start_positions)
