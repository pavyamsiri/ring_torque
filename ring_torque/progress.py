"""Custom progress bar that shows the speed of the task."""

# Internal libraries
import time

# External libraries
from rich.panel import Panel
from rich.progress import Progress
from rich import progress as rprogress


class CustomProgress(Progress):
    """Custom progress bar that shows the speed of the task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.perf_counter()

    def get_renderables(self):
        for task in self.tasks:
            elapsed_time = time.perf_counter() - self.start_time
            speed = task.completed / elapsed_time
            self.columns = (
                rprogress.SpinnerColumn(),
                *Progress.get_default_columns(),
                rprogress.TimeElapsedColumn(),
                rprogress.TextColumn(f"Speed: {speed:.2f} it/s"),
                rprogress.TextColumn(f"{task.completed}/{task.total}"),
            )
            for renderable in super().get_renderables():
                yield Panel(renderable, expand=False)
