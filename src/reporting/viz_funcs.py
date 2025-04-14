from ax.plot.slice import plot_slice_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import polars as pl


def plot_slice_ax(ax_client, param_name, objective_name):
    return plot_slice_plotly(
        ax_client.generation_strategy.model, param_name, objective_name
    )


def plot_powerband_distributions(df, columns, partition_col):
    # Create subplot for each column
    fig = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)

    # Get unique partition values - convert to list for polars
    # Get top 2 partitions by count
    partitions = (
        df.group_by(partition_col)
        .agg(pl.count())
        .sort(pl.col("count"), descending=True)
        .head(2)
        .get_column(partition_col)
        .to_list()
    )

    # Define KDE parameters
    kde_points = 200

    # Color scale for different partitions
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot KDE for each column and partition
    for col_idx, col in enumerate(columns):
        for part_idx, partition in enumerate(partitions):
            # Get data for this partition using polars filtering
            data = (
                df.filter(pl.col(partition_col) == partition)
                .get_column(col)
                .drop_nulls()
            )

            if len(data) > 1:  # Need at least 2 points for KDE
                # Convert polars series to numpy array for KDE calculation
                data_np = data.to_numpy()

                # Calculate KDE
                kde = stats.gaussian_kde(data_np)
                x_range = np.linspace(min(data_np), max(data_np), kde_points)
                y_range = kde(x_range)

                # Add trace to subplot
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        name=f"{partition}",
                        showlegend=col_idx == 0,  # Show legend only for first column
                        line=dict(color=colors[part_idx % len(colors)]),
                    ),
                    row=1,
                    col=col_idx + 1,
                )

    # Update layout
    fig.update_layout(
        height=400,
        width=300 * len(columns),
        title_text="Distribution of Power Bands",
        showlegend=True,
    )

    # Update y-axes titles
    for i in range(len(columns)):
        fig.update_yaxes(title_text="Density", row=1, col=i + 1)
        fig.update_xaxes(title_text="Value", row=1, col=i + 1)

    return fig
