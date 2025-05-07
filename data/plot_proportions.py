import pandas as pd
import matplotlib.pyplot as plt

def plot_pie_with_legend_and_hatch(csv_file, value_column, count_column, title="Pie Chart"):
    """
    Plots a pie chart from a CSV file with legend on the side and hatches.

    Args:
        csv_file (str): Path to the CSV file.
        value_column (str): Name of the column containing the values.
        count_column (str): Name of the column containing the counts.
        title (str, optional): Title of the pie chart. Defaults to "Pie Chart".
    """
    try:
        df = pd.read_csv(csv_file)

        if value_column not in df.columns or count_column not in df.columns:
            print(f"Error: One or both of the specified columns not found in the CSV file.")
            return

        values = df[value_column]
        counts = df[count_column]

        plt.figure(figsize=(10, 8), dpi=300)  # Adjust figure size for legend

        # Define hatches and colors
        hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.']
        colors = plt.cm.get_cmap('Set3').colors  # Get a colormap

        patches, texts, autotexts = plt.pie(
            counts,
            autopct='%1.1f%%',
            startangle=110,
            colors=colors[:len(values)],  # Use colors from colormap
            pctdistance=1.2,
            textprops={'fontsize': 15}
        )

        # Apply hatches
        for i, patch in enumerate(patches):
            patch.set_hatch(hatches[i % len(hatches)])

        # Add legend on the side
        plt.legend(
            patches,
            values,
            title="Cancer type",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            # textprops={'fontsize': 14}
            prop={'size': 12}
        )

        # plt.title(title)
        plt.axis('equal')
        plt.tight_layout() # to make room for legend
        plt.show()

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


plot_pie_with_legend_and_hatch("proportions_by_type.csv", "Type", "Proportion", "Cancer proportion by type")