"""Interactive English demonstration of the proposed allocation method."""

import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

from algorithm import generate_local_bundle_preview, run_energy_constrained_bundle
from data_scenario import RealDataScenarioGenerator
from experiment_config import DATA_PATH


EXECUTION_LOADS = {"I": 6, "M": 5, "S": 3, "N": 2}
TASK_COLOURS = {
    "I": "#e74c3c",
    "M": "#f39c12",
    "S": "#8e63c7",
    "N": "#8c6d62",
}
UAV_COLOURS = [
    "#1769aa",
    "#00a878",
    "#d94f9b",
    "#7654b8",
    "#008f9c",
    "#c56900",
    "#56616f",
    "#b52c35",
    "#16776b",
    "#7041ba",
]


class DemonstrationWindow:
    def __init__(self, root):
        self.root = root
        self.font_family = "Arial"
        self.root.title("UAV task allocation demonstration")
        self.root.geometry("1680x960")
        self.root.minsize(1400, 820)
        self.root.tk.call("tk", "scaling", 1.25)
        self.root.configure(bg="#f4f5f7")

        self.generator = RealDataScenarioGenerator(DATA_PATH, grid_size=64)
        self.task_limit = min(40, len(self.generator.data))
        self.seed = 20260729
        self.auto_playing = False
        self.scheduled_action = None
        self.synchronising_controls = False
        self.task_count_variables = {
            task_type: tk.IntVar(value=5) for task_type in "IMSN"
        }
        self.uav_count = tk.IntVar(value=5)
        self.battery_capacity = tk.IntVar(value=160)
        self.total_task_text = tk.StringVar()
        self.stage_text = tk.StringVar()
        self.summary_text = tk.StringVar()
        self.stage = "needs_map"
        self.scenario = None
        self.current_positions = []
        self.remaining_energy = []
        self.completed_task_ids = set()
        self.executed_paths = []
        self.local_details = None
        self.final_details = None

        self._build_interface()
        self._bind_parameter_changes()
        self.generate_scenario(use_next_seed=False)

    def _build_interface(self):
        title_bar = tk.Frame(
            self.root,
            bg="#ffffff",
            height=72,
            highlightbackground="#202124",
            highlightthickness=2,
        )
        title_bar.pack(fill=tk.X, padx=12, pady=(12, 0))
        title_bar.pack_propagate(False)
        tk.Label(
            title_bar,
            text="UAV task allocation demonstration",
            bg="#ffffff",
            fg="#202124",
            font=(self.font_family, 24, "bold"),
        ).pack(side=tk.LEFT, padx=26)
        tk.Label(
            title_bar,
            textvariable=self.stage_text,
            bg="#e8f1fb",
            fg="#124e8c",
            font=(self.font_family, 16, "bold"),
            padx=18,
            pady=11,
        ).pack(side=tk.RIGHT, padx=18)

        main_area = tk.Frame(self.root, bg="#f4f5f7")
        main_area.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        main_area.grid_rowconfigure(0, weight=1)
        main_area.grid_columnconfigure(0, weight=30, minsize=450)
        main_area.grid_columnconfigure(1, weight=70, minsize=850)

        self.left_panel = tk.Frame(
            main_area,
            bg="#ffffff",
            highlightbackground="#202124",
            highlightthickness=2,
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        map_panel = tk.Frame(
            main_area,
            bg="#ffffff",
            highlightbackground="#202124",
            highlightthickness=2,
        )
        map_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self.settings_panel = tk.Frame(self.left_panel, bg="#ffffff")
        self.settings_panel.pack(fill=tk.X)
        self.result_panel = tk.Frame(
            self.left_panel,
            bg="#ffffff",
            highlightbackground="#202124",
            highlightthickness=2,
        )
        self.result_panel.pack(fill=tk.BOTH, expand=True, padx=14, pady=(16, 14))
        self._build_controls()
        self._build_summary()
        self._build_map(map_panel)
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)

    def _section_heading(self, parent, text):
        tk.Label(
            parent,
            text=text,
            bg="#ffffff",
            fg="#202124",
            anchor="w",
            font=(self.font_family, 18, "bold"),
        ).pack(fill=tk.X, padx=22, pady=(20, 10))

    def _build_controls(self):
        self._section_heading(self.settings_panel, "Scenario settings")
        for task_type in "IMSN":
            self._slider(
                self.settings_panel,
                f"Task type {task_type}   load {EXECUTION_LOADS[task_type]}",
                self.task_count_variables[task_type],
                0,
                self.task_limit,
                1,
            )
        tk.Label(
            self.settings_panel,
            textvariable=self.total_task_text,
            bg="#ffffff",
            fg="#202124",
            font=(self.font_family, 16, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=22, pady=(6, 6))
        tk.Label(
            self.settings_panel,
            text=(
                "For clarity, the demonstration uses 20 tasks and 5 UAVs by "
                "default. The formal experiments use 200 tasks and 10 UAVs."
            ),
            bg="#ffffff",
            fg="#5f6368",
            font=(self.font_family, 11),
            justify=tk.LEFT,
            anchor="w",
            wraplength=390,
        ).pack(fill=tk.X, padx=22, pady=(0, 12))
        self._slider(
            self.settings_panel,
            "Number of UAVs",
            self.uav_count,
            2,
            10,
            1,
        )
        self._slider(
            self.settings_panel,
            "Initial battery capacity per UAV",
            self.battery_capacity,
            40,
            240,
            5,
        )
        tk.Frame(self.settings_panel, bg="#202124", height=2).pack(
            fill=tk.X, padx=18, pady=12
        )

        row = tk.Frame(self.settings_panel, bg="#ffffff")
        row.pack(fill=tk.X, padx=18)
        self.new_map_button = self._button(
            row, "Generate a new map", self.generate_new_map
        )
        self.new_map_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        self.reset_button = self._button(
            row, "Reset this map", self.reset_current_scenario
        )
        self.reset_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )

        self.bundle_button = self._button(
            self.settings_panel,
            "1. Build local task bundles",
            self.build_local_task_bundles,
        )
        self.bundle_button.pack(fill=tk.X, padx=18, pady=(12, 4))
        self.consensus_button = self._button(
            self.settings_panel,
            "2. Exchange results and resolve conflicts",
            self.confirm_consensus,
        )
        self.consensus_button.pack(fill=tk.X, padx=18, pady=4)
        self.execute_button = self._button(
            self.settings_panel,
            "3. Execute the agreed paths",
            self.execute_tasks,
        )
        self.execute_button.pack(fill=tk.X, padx=18, pady=4)
        self.auto_button = self._button(
            self.settings_panel, "Run automatically", self.toggle_auto_play
        )
        self.auto_button.pack(fill=tk.X, padx=18, pady=(10, 4))

    def _slider(self, parent, label, variable, minimum, maximum, step):
        row = tk.Frame(parent, bg="#ffffff")
        row.pack(fill=tk.X, padx=22, pady=(10, 1))
        tk.Label(
            row,
            text=label,
            bg="#ffffff",
            fg="#202124",
            font=(self.font_family, 15, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            row,
            textvariable=variable,
            bg="#ffffff",
            fg="#202124",
            font=(self.font_family, 18, "bold"),
            width=3,
        ).pack(side=tk.RIGHT)
        tk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            resolution=step,
            orient=tk.HORIZONTAL,
            variable=variable,
            showvalue=False,
            bg="#ffffff",
            fg="#202124",
            troughcolor="#d4d7db",
            activebackground="#3c4043",
            highlightthickness=0,
            sliderrelief=tk.RAISED,
            bd=0,
            length=345,
            width=22,
        ).pack(anchor="w", padx=22)

    def _button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#f7f7f7",
            fg="#202124",
            activebackground="#eeeeee",
            font=(self.font_family, 16, "bold"),
            relief=tk.FLAT,
            bd=0,
            highlightbackground="#202124",
            highlightthickness=2,
            pady=12,
            cursor="hand2",
        )

    def _build_summary(self):
        self._section_heading(self.result_panel, "Current result")
        tk.Frame(self.result_panel, bg="#202124", height=2).pack(
            fill=tk.X, padx=20, pady=(0, 14)
        )
        tk.Label(
            self.result_panel,
            textvariable=self.summary_text,
            bg="#ffffff",
            fg="#202124",
            justify=tk.LEFT,
            anchor="nw",
            wraplength=390,
            font=(self.font_family, 16),
            padx=22,
        ).pack(fill=tk.BOTH, expand=True, pady=(2, 14))

    def _build_map(self, parent):
        self.figure, self.axis = plt.subplots(figsize=(8.1, 7.1))
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=8, pady=8
        )

    def _bind_parameter_changes(self):
        variables = list(self.task_count_variables.values()) + [
            self.uav_count,
            self.battery_capacity,
        ]
        for variable in variables:
            variable.trace_add("write", self.parameter_changed)

    def parameter_changed(self, *_):
        if self.synchronising_controls:
            return
        total_tasks = sum(
            variable.get() for variable in self.task_count_variables.values()
        )
        if total_tasks > self.task_limit:
            for variable in reversed(list(self.task_count_variables.values())):
                if total_tasks <= self.task_limit:
                    break
                reduction = min(
                    variable.get(), total_tasks - self.task_limit
                )
                self.synchronising_controls = True
                variable.set(variable.get() - reduction)
                self.synchronising_controls = False
                total_tasks -= reduction
        if total_tasks < 1:
            self.synchronising_controls = True
            self.task_count_variables["I"].set(1)
            self.synchronising_controls = False
            total_tasks = 1

        self.total_task_text.set(
            f"Total tasks: {total_tasks}, maximum {self.task_limit}"
        )
        if self.scenario is not None and self.stage not in (
            "needs_map",
            "finished",
        ):
            self._stop_auto_play()
            self.stage = "needs_map"
            self.summary_text.set(
                "The settings have changed. Generate a new map to continue."
            )
            self._update_buttons()
            self._redraw()

    def _current_task_composition(self):
        return {
            task_type: variable.get()
            for task_type, variable in self.task_count_variables.items()
        }

    def generate_scenario(self, use_next_seed=True):
        self._stop_auto_play()
        if use_next_seed:
            self.seed += 1
        try:
            self.scenario = self.generator.generate(
                self.seed,
                self._current_task_composition(),
                maximum_uav_count=self.uav_count.get(),
            )
        except Exception as error:
            self.summary_text.set(str(error))
            return
        self._reset_runtime_state()
        self.summary_text.set(
            "The map is ready. Select step 1 to build a local bundle for each UAV."
        )
        self._update_buttons()
        self._redraw()

    def generate_new_map(self):
        self.generate_scenario(use_next_seed=True)

    def _reset_runtime_state(self):
        start_positions = list(
            self.scenario.uav_start_positions[: self.uav_count.get()]
        )
        self.current_positions = list(start_positions)
        self.remaining_energy = [
            float(self.battery_capacity.get()) for _ in start_positions
        ]
        self.completed_task_ids = set()
        self.executed_paths = [[position] for position in start_positions]
        self.local_details = None
        self.final_details = None
        self.stage = "ready"

    def reset_current_scenario(self):
        if self.scenario is None:
            self.generate_scenario(use_next_seed=False)
            return
        self._stop_auto_play()
        self._reset_runtime_state()
        self.summary_text.set(
            "The scenario has been reset. All UAVs are back at their starting positions."
        )
        self._update_buttons()
        self._redraw()

    def build_local_task_bundles(self):
        if self.stage != "ready":
            return
        self.local_details = generate_local_bundle_preview(
            self.scenario,
            self.uav_count.get(),
            self.battery_capacity.get(),
            EXECUTION_LOADS,
        )
        self.stage = "bundles_built"
        lines = []
        for uav_id, (route, cost) in enumerate(
            zip(
                self.local_details["routes_by_uav"],
                self.local_details["route_costs_by_uav"],
            )
        ):
            route_text = ", ".join(
                f"T{task_id + 1}" for task_id in route[:6]
            )
            if len(route) > 6:
                route_text += ", ..."
            lines.append(
                f"UAV {uav_id + 1}: {len(route)} tasks, cost {cost:.1f}\n"
                f"{route_text}"
            )
        self.summary_text.set(
            "Each UAV builds a bundle using local information:\n\n"
            + "\n\n".join(lines)
            + "\n\nThe local bundles may conflict. The next step exchanges "
            "winner tables."
        )
        self._update_buttons()
        self._redraw()
        self._schedule_auto_play()

    def confirm_consensus(self):
        if self.stage != "bundles_built":
            return
        self.final_details = run_energy_constrained_bundle(
            self.scenario,
            self.uav_count.get(),
            self.battery_capacity.get(),
            EXECUTION_LOADS,
            return_details=True,
        )
        self.stage = "consensus_confirmed"
        lines = []
        for uav_id, (route, cost) in enumerate(
            zip(
                self.final_details["routes_by_uav"],
                self.final_details["costs_by_uav"],
            )
        ):
            lines.append(
                f"UAV {uav_id + 1}: {len(route)} assigned tasks, "
                f"path cost {cost:.1f}"
            )
        self.summary_text.set(
            "Conflicts are resolved and the local winner and bid tables agree:\n\n"
            + "\n".join(lines)
            + "\n\nThe UAVs can now execute the agreed paths."
        )
        self._update_buttons()
        self._redraw()
        self._schedule_auto_play()

    def execute_tasks(self):
        if self.stage != "consensus_confirmed" or self.final_details is None:
            return
        task_table = {task.task_id: task for task in self.scenario.tasks}
        for uav_id, route_ids in enumerate(
            self.final_details["routes_by_uav"]
        ):
            current_position = self.current_positions[uav_id]
            for task_id in route_ids:
                task = task_table[task_id]
                cost = abs(current_position[0] - task.coordinate[0]) + abs(
                    current_position[1] - task.coordinate[1]
                )
                cost += EXECUTION_LOADS[task.task_type]
                self.remaining_energy[uav_id] -= cost
                current_position = task.coordinate
                self.executed_paths[uav_id].append(current_position)
                self.completed_task_ids.add(task_id)
            self.current_positions[uav_id] = current_position
        self.stage = "finished"
        self._stop_auto_play()
        self.summary_text.set(
            "Demonstration finished.\n"
            f"Completed tasks: {len(self.completed_task_ids)}/"
            f"{len(self.scenario.tasks)}\n"
            "Every UAV bundle satisfies its individual energy limit."
        )
        self._update_buttons()
        self._redraw()

    def advance(self):
        if self.stage == "ready":
            self.build_local_task_bundles()
        elif self.stage == "bundles_built":
            self.confirm_consensus()
        elif self.stage == "consensus_confirmed":
            self.execute_tasks()

    def toggle_auto_play(self):
        if self.stage in ("needs_map", "finished"):
            return
        self.auto_playing = not self.auto_playing
        if self.auto_playing:
            self.auto_button.configure(text="Pause")
            self.advance()
        else:
            self._stop_auto_play()

    def _schedule_auto_play(self):
        if self.auto_playing and self.stage not in ("needs_map", "finished"):
            self.scheduled_action = self.root.after(1000, self.advance)

    def _stop_auto_play(self):
        self.auto_playing = False
        if self.scheduled_action is not None:
            try:
                self.root.after_cancel(self.scheduled_action)
            except tk.TclError:
                pass
        self.scheduled_action = None

    def close_window(self):
        self._stop_auto_play()
        self.root.destroy()

    def _update_buttons(self):
        button_states = {
            "ready": (tk.NORMAL, tk.DISABLED, tk.DISABLED),
            "bundles_built": (tk.DISABLED, tk.NORMAL, tk.DISABLED),
            "consensus_confirmed": (tk.DISABLED, tk.DISABLED, tk.NORMAL),
            "needs_map": (tk.DISABLED, tk.DISABLED, tk.DISABLED),
            "finished": (tk.DISABLED, tk.DISABLED, tk.DISABLED),
        }
        for button, state in zip(
            (self.bundle_button, self.consensus_button, self.execute_button),
            button_states[self.stage],
        ):
            button.configure(state=state)
        stage_labels = {
            "ready": "Ready to build local bundles",
            "bundles_built": "Local task bundles are ready",
            "consensus_confirmed": "Task conflicts have been resolved",
            "needs_map": "Generate a new map",
            "finished": "Demonstration finished",
        }
        self.stage_text.set(stage_labels[self.stage])
        self.auto_button.configure(
            text="Pause" if self.auto_playing else "Run automatically"
        )

    def _route_coordinates(self, route_ids, start):
        task_table = {task.task_id: task for task in self.scenario.tasks}
        return [start] + [
            task_table[task_id].coordinate for task_id in route_ids
        ]

    def _redraw(self):
        axis = self.axis
        axis.clear()
        axis.set_facecolor("#ffffff")
        if self.scenario is None:
            self.canvas.draw_idle()
            return

        local_routes = (
            self.local_details["routes_by_uav"] if self.local_details else ()
        )
        final_routes = (
            self.final_details["routes_by_uav"] if self.final_details else ()
        )
        local_occurrences = {}
        for route in local_routes:
            for task_id in route:
                local_occurrences[task_id] = local_occurrences.get(task_id, 0) + 1
        conflicting_tasks = {
            task_id
            for task_id, count in local_occurrences.items()
            if count > 1
        }
        important_tasks = set(conflicting_tasks)
        for route in final_routes:
            important_tasks.update(route)
        show_all_labels = len(self.scenario.tasks) <= 25

        for task in self.scenario.tasks:
            completed = task.task_id in self.completed_task_ids
            axis.scatter(
                *task.coordinate,
                s=115,
                color=TASK_COLOURS[task.task_type],
                alpha=0.22 if completed else 1.0,
                edgecolors="#202124",
                linewidths=1.3,
                zorder=2,
            )
            if task.task_id in conflicting_tasks and self.stage == "bundles_built":
                axis.scatter(
                    *task.coordinate,
                    s=250,
                    facecolors="none",
                    edgecolors="#d4a017",
                    linewidths=3.0,
                    zorder=5,
                )
            if show_all_labels or task.task_id in important_tasks:
                label = (
                    f"✓ T{task.task_id + 1}"
                    if completed
                    else f"T{task.task_id + 1}"
                )
                axis.annotate(
                    label,
                    task.coordinate,
                    xytext=(7, 6),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color="#555555",
                )

        for uav_id, start in enumerate(
            self.scenario.uav_start_positions[: self.uav_count.get()]
        ):
            colour = UAV_COLOURS[uav_id % len(UAV_COLOURS)]
            axis.scatter(
                *start,
                s=95,
                marker="s",
                facecolors="none",
                edgecolors=colour,
                linewidths=2.4,
                zorder=3,
            )

        if self.stage == "bundles_built":
            routes_to_draw = local_routes
            line_width = 2.0
            alpha = 0.65
        elif self.stage in ("consensus_confirmed", "finished"):
            routes_to_draw = final_routes
            line_width = 3.0
            alpha = 1.0
        else:
            routes_to_draw = ()
            line_width = 0.0
            alpha = 1.0

        for uav_id, route_ids in enumerate(routes_to_draw):
            coordinates = self._route_coordinates(
                route_ids, self.scenario.uav_start_positions[uav_id]
            )
            if len(coordinates) > 1:
                axis.plot(
                    [point[0] for point in coordinates],
                    [point[1] for point in coordinates],
                    color=UAV_COLOURS[uav_id % len(UAV_COLOURS)],
                    linewidth=line_width,
                    alpha=alpha,
                    zorder=3,
                )

        for uav_id, position in enumerate(self.current_positions):
            colour = UAV_COLOURS[uav_id % len(UAV_COLOURS)]
            axis.scatter(
                *position,
                s=145,
                marker="D",
                color=colour,
                edgecolors="#202124",
                linewidths=1.2,
                zorder=6,
            )
            axis.annotate(
                f"U{uav_id + 1}\n{self.remaining_energy[uav_id]:.0f}",
                position,
                xytext=(9, -23),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color=colour,
            )

        axis.set_title(
            "Proposed method task allocation process",
            fontsize=20,
            fontweight="bold",
            pad=14,
        )
        axis.set_xlabel("Grid x coordinate", fontsize=14, fontweight="bold")
        axis.set_ylabel("Grid y coordinate", fontsize=14, fontweight="bold")
        axis.tick_params(labelsize=12)
        axis.grid(True, linewidth=0.8, alpha=0.28)
        axis.set_aspect("equal", adjustable="datalim")
        for border in axis.spines.values():
            border.set_linewidth(1.8)
            border.set_color("#202124")

        legend_items = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=TASK_COLOURS[task_type],
                markeredgecolor="#202124",
                label=(
                    f"Task type {task_type}  load {EXECUTION_LOADS[task_type]}"
                ),
                markersize=11,
            )
            for task_type in "IMSN"
        ]
        legend_items += [
            Line2D(
                [0],
                [0],
                marker="s",
                color=UAV_COLOURS[0],
                markerfacecolor="none",
                label="UAV starting position",
                markersize=11,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=UAV_COLOURS[0],
                label="UAV current position",
                markersize=11,
            ),
        ]
        axis.legend(
            handles=legend_items,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            frameon=True,
            edgecolor="#202124",
            fontsize=13,
            borderpad=0.8,
            labelspacing=0.7,
            handletextpad=0.6,
        )
        self.figure.tight_layout(rect=(0, 0.10, 1, 1))
        self.canvas.draw_idle()


def main():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    root = tk.Tk()
    DemonstrationWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
