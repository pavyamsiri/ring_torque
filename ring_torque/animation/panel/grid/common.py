# Standard libraries
from typing import Sequence

# External libraries
import numpy as np
from numpy import typing as npt

# Internal libraries

DEFAULT_NUM_GRID_CELLS: int = 128


def get_all_neighbours(
    grid: npt.NDArray[np.floating], x_idx: int, y_idx: int
) -> Sequence[float]:
    """Returns a list of all the neighbouring values of a cell in a grid.
    Neighbours are defined as the cells to the left, right, above, below, top left, top right, bottom left and bottom.

    Parameters
    ----------
    grid : npt.NDArray[np.floating]
        The 2x2 grid to get the neighbours from.
    x_idx : int
        The x index of the cell.
    y_idx : int
        The y index of the cell.

    Returns
    -------
    Sequence[float]
        A list of all the neighbouring values of a cell in a grid.
    """
    num_cells = grid.shape[0]
    assert grid.shape == (num_cells, num_cells)
    values: Sequence[float] = []
    for row_offset in range(-1, 2):
        for col_offset in range(-1, 2):
            if row_offset == 0 and col_offset == 0:
                continue
            row = y_idx + row_offset
            col = x_idx + col_offset
            if row < 0 or row >= num_cells or col < 0 or col >= num_cells:
                continue
            values.append(float(grid[row, col]))
    return values


def interpolate_grid(
    grid: npt.NDArray[np.floating], max_trials: int = 10000
) -> npt.NDArray[np.floating]:
    """Interpolates a grid by filling in the missing values with the mean of its neighbours.

    Parameters
    ----------
    grid : npt.NDArray[np.floating]
        The 2x2 grid to interpolate.
    max_trials : int, optional
        The maximum number of trials to attempt to interpolate the grid. Defaults to 10000.

    Returns
    -------
    npt.NDArray[np.floating]
        The interpolated grid.
    """
    num_cells = grid.shape[0]
    assert grid.shape == (num_cells, num_cells)
    # Create a work queue of all the cells that need to be interpolated
    work_queue = []
    for x_idx in range(num_cells):
        for y_idx in range(num_cells):
            if np.isnan(grid[y_idx, x_idx]):
                values = get_all_neighbours(grid, x_idx, y_idx)
                if np.all(np.isnan(values)):
                    continue
                work_queue.append((x_idx, y_idx))

    # While work queue is non-empty attempt to interpolate
    count = 0
    while len(work_queue) > 0 and count < max_trials:
        count += 1
        x_idx, y_idx = work_queue.pop(0)
        values = get_all_neighbours(grid, x_idx, y_idx)
        new_value = np.nanmean(values)
        # If the new value is nan and there are some non-nan neighbours, add it back to the work queue
        if np.isnan(new_value) and not np.all(np.isnan(values)):
            work_queue.append((x_idx, y_idx))
        # Otherwise, update the grid
        elif not np.isnan(new_value):
            grid[y_idx, x_idx] = new_value
    return grid
