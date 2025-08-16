
import numpy as np


def plot_slices(sinogram, n_slices=None, x_extent=None, y_extent=None, x_unit='', y_unit='', bar='counts', rows=4, cols=4, colormap='jet', fontsize=12, start_slice=0, end_slice=None,folder_name="",file_name=""):
    """
    Plots slices of a 3D sinogram in a grid, creating new figures when the number of slices exceeds the grid size.
    Allows setting the extent for x-axis and y-axis, and displays the axes values respecting the specified extents.
    Also allows specifying the range of slices to plot.

    Parameters:
    sinogram (numpy.ndarray): A 3D numpy array of shape (slices, distances, angles).
    n_slices (int): The number of slices to plot. If None, will plot all slices.
    x_extent (tuple): A tuple specifying the x-axis extent (min, max).
    y_extent (tuple): A tuple specifying the y-axis extent (min, max).
    x_unit (str): The unit for the x-axis.
    y_unit (str): The unit for the y-axis.
    rows (int): The number of rows in the subplot grid.
    cols (int): The number of columns in the subplot grid.
    fontsize (int): The font size for axis and colorbar labels.
    start_slice (int): The index of the first slice to plot (inclusive).
    end_slice (int): The index of the last slice to plot (exclusive). If None, will plot up to the last slice.
    """
    import matplotlib.pyplot as plt
    # Check the shape of the sinogram
    if len(sinogram.shape) != 3:
        raise ValueError("Input sinogram must be a 3D numpy array.")

    slices, distances, angles = sinogram.shape

    # Determine the number of slices to plot
    if n_slices is None or n_slices > slices:
        n_slices = slices

    # Adjust the start and end slice indices based on the specified range
    start_slice = max(0, start_slice)
    end_slice = min(slices, end_slice) if end_slice is not None else slices

    # Calculate the number of figures needed
    n_figures = (end_slice - start_slice - 1) // (rows * cols) + 1
    counter = 1

    # Create the figures and plot the slices
    for fig_idx in range(n_figures):
        fig_start_slice = start_slice + fig_idx * (rows * cols)
        fig_end_slice = min(fig_start_slice + (rows * cols), end_slice)

        # Create a figure for plotting
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Plot each slice
        for i in range(fig_start_slice, fig_end_slice):
            im = axes[i - fig_start_slice].imshow(sinogram[i], cmap=colormap, aspect='auto',
                                                  extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]) if x_extent and y_extent else None)
            axes[i - fig_start_slice].set_title(f'Slice {i + 1}')

            # Set x-axis and y-axis labels with units and font size
            axes[i - fig_start_slice].set_xlabel(f'{x_unit}', fontsize=fontsize)
            axes[i - fig_start_slice].set_ylabel(f'{y_unit}', fontsize=fontsize)

            # Add a colorbar to each subplot
            cbar = fig.colorbar(im, ax=axes[i - fig_start_slice], orientation='vertical')
            cbar.ax.tick_params(labelsize=fontsize)  # Set font size for colorbar ticks
            cbar.set_label(bar, fontsize=fontsize)  # Set font size for colorbar label

        # Hide any unused subplots
        for j in range(fig_end_slice - fig_start_slice, rows * cols):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(f"{folder_name}/{file_name}{counter}.png")
        counter += 1
        # plt.show()