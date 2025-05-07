import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import pandas as pd
import os
import toml

# Path to the main directory containing the result folders
main_directory = './results'

# Name of the statistics file
stats_file_name = 'solution_stats.csv'
kpi_file_prefix = 'kpis_instance_'

# Extension of the configuration file
config_file_extension = '.toml'
kpi_file_extension = '.csv'

# List to store the aggregated data for each instance
results_list = []

# Names of the columns for which to calculate the cumulative sum
column1_name = 'objective_first_stage_earliest'  # Replace with the actual name
column2_name = 'objective_first_stage_treatment_range'   # Replace with the actual name
gap_column_name = 'gap'
kpi_treatment_dates_col = 'sum_qi_treatment_dates'
kpi_count_violated_col = 'sum_qi_count_violated'

# Iterate through all items in the main directory
for item in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, item)

    # Check if the item is a directory
    if os.path.isdir(folder_path):
        config_file_name = "config_"+ item + config_file_extension
        config_file_path = os.path.join(folder_path, config_file_name)
        stats_file_path = os.path.join(folder_path, stats_file_name)
        kpi_file_name = f"{kpi_file_prefix}{kpi_file_extension}"
        kpi_file_path = os.path.join(folder_path, kpi_file_name)

        # Initialize a dictionary to store the results for this instance
        instance_results = {}

        # print(item.split('_'))
        # Extract hashes from the folder name
        try:
            _, instance_hash, config_hash = item.split('_')
            instance_results['instance_hash'] = instance_hash
            # instance_results['config_hash'] = config_hash
        except ValueError:
            instance_results['instance_hash'] = item
            # instance_results['config_hash'] = None
            print(f"Warning: Folder name '{item}' does not conform to the expected format.")

        # Read the TOML configuration file if it exists
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    config_data = toml.load(f)
                # Add configuration parameters to the results dictionary
                if 'general' in config_data:
                    instance_results['config_hash_toml'] = config_data['general'].get('config_hash')
                if 'simulation' in config_data:
                    instance_results['instance_data_file'] = config_data['simulation'].get('instance_data_file')
                    instance_results['scenarios'] = config_data['simulation'].get('scenarios')
                    instance_results['aggregation_criterion'] = config_data['simulation'].get('aggregation_criterion')
                    instance_results['first_stage_weight_proportion'] = config_data['simulation'].get('first_stage_weight_proportion')
                    instance_results['queue_length'] = config_data['simulation'].get('queue_length')
                    instance_results['one_scenario_strategy'] = config_data['simulation'].get('one_scenario_strategy')
                print(f"TOML configuration read for '{folder_path}'.")
            except Exception as e:
                print(f"Error reading TOML configuration file '{config_file_path}': {e}")
        else:
            print(f"Configuration file '{config_file_name}' is absent from the folder '{folder_path}'.")

        # Process the statistics file if it exists
        if os.path.exists(stats_file_path):
            try:
                stats_df = pd.read_csv(stats_file_path)
                if column1_name in stats_df.columns and column2_name in stats_df.columns:
                    cumulative_sum_col1 = stats_df[column1_name][10:].cumsum().iloc[-1]
                    cumulative_sum_col2 = stats_df[column2_name][10:].cumsum().iloc[-1]
                    mean_gap = stats_df[gap_column_name][10:].mean()

                    instance_results[f'cumulative_sum_{column1_name}'] = cumulative_sum_col1
                    instance_results[f'cumulative_sum_{column2_name}'] = cumulative_sum_col2
                    instance_results[f'cumulative_sum_objective_first_stage'] = cumulative_sum_col1 + cumulative_sum_col2
                    instance_results[f'mean_{gap_column_name}'] = mean_gap
                    # instance_results[f'cumulative_sum_{column2_name}'] = cumulative_sum_col2
                    results_list.append(instance_results)
                    print(f"Cumulative sums calculated for '{stats_file_path}'.")
                else:
                    print(f"Warning: Columns '{column1_name}' or '{column2_name}' are missing from '{stats_file_path}'.")
            except Exception as e:
                print(f"Error processing '{stats_file_path}': {e}")
        else:
            print(f"The file '{stats_file_name}' is absent from the folder '{folder_path}'.")

        # Traiter le fichier KPI s'il existe
        if os.path.exists(kpi_file_path):
            try:
                kpi_df = pd.read_csv(kpi_file_path)
                if kpi_treatment_dates_col in kpi_df.columns and kpi_count_violated_col in kpi_df.columns and len(
                        kpi_df) >= 2:
                    last_treatment_dates = kpi_df[kpi_treatment_dates_col].iloc[-1]
                    last_count_violated = kpi_df[kpi_count_violated_col].iloc[-1]

                    instance_results['kpi_waiting_times'] = last_treatment_dates
                    instance_results['kpi_count_violated'] = last_count_violated
                    results_list.append(instance_results)
                    print(f"Extracted KPI data from '{kpi_file_path}'.")
                else:
                    print(
                        f"Warning: The file '{kpi_file_path}' does not contain the required columns ('{kpi_treatment_dates_col}', '{kpi_count_violated_col}') or has less than 2 rows.")
            except Exception as e:
                print(f"Error processing KPI file '{kpi_file_path}': {e}")
        else:
            print(f"The KPI file '{kpi_file_name}' is absent from the folder '{folder_path}'.")

# Create the final DataFrame
if results_list:
    grand_dataframe = pd.DataFrame(results_list)
    # grand_dataframe.to_csv("simulations_res.csv", index=False)
else:
    print("No data to include in the DataFrame.")

objective_column = 'cumulative_sum_objective_first_stage'
grand_dataframe['difference_to_deterministic'] = pd.NA
grand_dataframe['kpi_diff_wt'] = pd.NA
grand_dataframe['kpi_diff_cv'] = pd.NA


for instance in grand_dataframe['instance_hash'].unique():
    instance_df = grand_dataframe[grand_dataframe['instance_hash'] == instance].copy()
    deterministic_value = instance_df.loc[instance_df['scenarios'] == 0, objective_column].iloc[0] if not instance_df[instance_df['scenarios'] == 0].empty else None
    deterministic_kpi_waiting_time = instance_df.loc[instance_df['scenarios'] == 0, "kpi_waiting_times"].iloc[0] if not instance_df[instance_df['scenarios'] == 0].empty else None
    deterministic_kpi_count_violated = instance_df.loc[instance_df['scenarios'] == 0, "kpi_count_violated"].iloc[0] if not instance_df[instance_df['scenarios'] == 0].empty else None
    if deterministic_value is not None :
        grand_dataframe.loc[grand_dataframe['instance_hash'] == instance, 'difference_to_deterministic'] = (
            (instance_df[objective_column] - deterministic_value)/deterministic_value*100
        ).values
        if deterministic_kpi_waiting_time == 0 and instance_df["kpi_waiting_times"] == 0:
            grand_dataframe.loc[grand_dataframe['instance_hash'] == instance, 'kpi_diff_wt'] = 0
        else :
            grand_dataframe.loc[grand_dataframe['instance_hash'] == instance, 'kpi_diff_wt'] = (
                    (instance_df["kpi_waiting_times"] - deterministic_kpi_waiting_time ) / deterministic_kpi_waiting_time  * 100
            ).values
        grand_dataframe.loc[grand_dataframe['instance_hash'] == instance, 'kpi_diff_cv'] = (
                (instance_df["kpi_count_violated"] - deterministic_kpi_count_violated) / deterministic_kpi_count_violated * 100
        ).values
    else:
        print(f"Warning: Aucune ligne avec scenarios = 0 trouvée pour l'instance '{instance}'.")

grand_dataframe.to_csv("simulations_res.csv", index=False)

# Group by 'instance_hash' and find the index of the row with the minimum 'difference_to_deterministic'
idx = grand_dataframe.groupby('instance_hash')['difference_to_deterministic'].idxmin()

# Select the 'instance_hash', 'scenarios', and 'difference_to_deterministic' columns
# for the rows where 'difference_to_deterministic' is minimum for each 'instance_hash'
min_difference_data = grand_dataframe.loc[idx, ['instance_hash', 'scenarios', 'difference_to_deterministic']]

# The result 'min_difference_data' will be a new DataFrame containing
# only the rows where 'difference_to_deterministic' is minimum for each
# 'instance_hash', and it will include the 'instance_hash', 'scenarios',
# and 'difference_to_deterministic' columns.

print(min_difference_data)

def dataframe_to_latex_table(df, caption="Minimum Difference Data", label="tab:min_difference"):
    """Converts a Pandas DataFrame to a LaTeX table environment."""
    num_cols = df.shape[1]
    latex_str = "\\begin{table}[h!]\n"
    latex_str += "    \\centering\n"
    latex_str += f"    \\caption{{{caption}}}\n"
    latex_str += f"    \\label{{{label}}}\n"
    latex_str += "    \\begin{tabular}{" + "c" * num_cols + "}\n"
    latex_str += "        \\hline\n"
    latex_str += "        " + " & ".join(df.columns.str.replace('_', r'\_')) + " \\\\\n"
    latex_str += "        \\hline\n"
    for i in range(df.shape[0]):
        latex_str += "        " + " & ".join(df.iloc[i].astype(str).str.replace('_', r'\_')) + " \\\\\n"
    latex_str += "        \\hline\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "\\end{table}\n"
    return latex_str

latex_table = dataframe_to_latex_table(min_difference_data)

print(latex_table)


# Choisissez la colonne numérique que vous souhaitez visualiser avec le box plot
column_to_visualize = 'cumulative_sum_objective_first_stage'  # Remplacez par le nom de la colonne numérique pertinente

# Filtrer le DataFrame pour obtenir les données où 'scenarios' est égal à 0
scenario_zero_data = grand_dataframe[grand_dataframe['scenarios'] == 0]

custom_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# # --- Créer le box plot ---
# plt.figure(figsize=(10, 6), dpi=300)  # Adjust figure size if needed
# sns.boxplot(x='instance_hash', y=column_to_visualize, data=grand_dataframe)
# # Ajouter les points rouges pour les cas où 'scenarios' est 0
# sns.scatterplot(x='instance_hash', y=column_to_visualize, data=scenario_zero_data, color='red', marker='o', s=50, zorder=10)
# plt.title(f'Box Plot of {column_to_visualize} by instance (red dot for scenario=0)')
# plt.xlabel('Instance')
# plt.ylabel(column_to_visualize)
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()
#
# # --- Création du Violin Plot ---
# plt.figure(figsize=(12, 7), dpi=300)
# sns.violinplot(x='instance_hash', y=column_to_visualize, data=grand_dataframe, inner=None, facecolor=None, edgecolor='black')
# sns.stripplot(x='instance_hash', y=column_to_visualize, hue='scenarios', data=grand_dataframe, dodge=True, size=5, jitter=0.2, zorder=10, palette=custom_palette)
# # sns.scatterplot(x='instance_hash', y=column_to_visualize, data=scenario_zero_data, color='red', marker='o', s=30, zorder=10)
# plt.title(f'Violin Plot of {column_to_visualize} by instance.')
# plt.xlabel('Instance')
# plt.ylabel(column_to_visualize)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.legend(title='Scenarios', loc='lower right') # Ajouter une légende
# plt.show()

# # --- Plot average gap boxplot by scenario ---
# plt.figure(figsize=(12, 7), dpi=300)
# sns.boxplot(x='scenarios', y='mean_gap', data=grand_dataframe)
# plt.title('Box Plot of average gap of simulation relative to scenarios number')
# plt.xlabel('Number of scenarios')
# plt.ylabel('Average gap (%)')
# plt.xticks(rotation=0)  # Rotation des étiquettes de l'axe x si nécessaire
# plt.tight_layout()
# plt.show()

# --- Plot diff boxplot by scenario ---
sub_df = grand_dataframe.loc[grand_dataframe['scenarios'] > 0]

fig, (ax1, ax2) = plt.subplots(1,2)
# plt.figure(figsize=(12, 7), dpi=300)
sns.boxplot(ax=ax1, x='scenarios', y='difference_to_deterministic', data=sub_df)
sns.boxplot(ax=ax2, x='scenarios', y='mean_gap', data=grand_dataframe)
# plt.title('Box Plot difference of first stage objective vs deterministic relative to scenarios number')
ax1.set_title('(a)')
ax1.set_xlabel('Number of scenarios')
ax1.set_ylabel('Difference to myopic model objective value (%)')
ax2.set_title('(b)')
ax2.set_xlabel('Number of scenarios')
ax2.set_ylabel('Average gap to best bound (%)')
# plt.xticks(rotation=0)  # Rotation des étiquettes de l'axe x si nécessaire
fig.tight_layout()
fig.set_size_inches(7, 4.8)
# plt.show()
plt.savefig('boxplot_diff_gap.pgf')

# --- Plot kpis boxplot by scenario ---]

fig, (ax1, ax2) = plt.subplots(1,2)
# plt.figure(figsize=(12, 7), dpi=300)
sns.boxplot(ax=ax1, x='scenarios', y='kpi_waiting_times', data=grand_dataframe)
sns.boxplot(ax=ax2, x='scenarios', y='kpi_count_violated', data=grand_dataframe)
# plt.title('Box Plot difference of first stage objective vs deterministic relative to scenarios number')
ax1.set_xlabel('Number of scenarios')
ax1.set_ylabel('Waiting days')
ax2.set_xlabel('Number of scenarios')
ax2.set_ylabel('Days of violation')
# plt.xticks(rotation=0)  # Rotation des étiquettes de l'axe x si nécessaire
fig.tight_layout()
fig.set_size_inches(7, 4)
# fig.show()
plt.savefig('boxplot_kpis.pgf')


# --- Plot 1s strategy boxplot ---
sub_df_1s = grand_dataframe.loc[grand_dataframe['scenarios'] == 1]
fig = plt.figure()
sns.boxplot(x='one_scenario_strategy', y='difference_to_deterministic', data=sub_df_1s)
# plt.title('Box Plot difference of first stage objective vs deterministic relative to one scenario strategy')
plt.xlabel('One scenario strategy')
plt.ylabel('Difference to myopic model objective value (%)')
plt.xticks(rotation=0)  # Rotation des étiquettes de l'axe x si nécessaire
plt.tight_layout()
# plt.show()
fig.set_size_inches(7, 4)

# Échapper les underscores dans les labels et le titre
current_xticklabels = [label.get_text().replace('_', r'\_') for label in plt.gca().get_xticklabels()]
plt.gca().set_xticklabels(current_xticklabels)
plt.ylabel(plt.gca().get_ylabel().replace('_', r'\_'))
plt.xlabel(plt.gca().get_xlabel().replace('_', r'\_'))
if plt.gca().get_title():
    plt.title(plt.gca().get_title().replace('_', r'\_'))

# Échapper les underscores dans la légende
legend = plt.gca().get_legend()
if legend:
    title = legend.get_title()
    if title:
        title.set_text(title.get_text().replace('_', r'\_'))
    for text in legend.get_texts():
        if hasattr(text, 'set_text'):
            text.set_text(text.get_text().replace('_', r'\_'))

plt.savefig('boxplot_one_scenario_strategy.pgf')

# --- Plot aggregation criteria ---
sub_df_agg = grand_dataframe.loc[grand_dataframe['scenarios'] > 1]
fig  = plt.figure()
sns.boxplot(x='scenarios', y='difference_to_deterministic', hue='aggregation_criterion', data=sub_df_agg)
# plt.title('Box Plot difference of first stage objective vs deterministic relative to one scenario strategy')
plt.xlabel('Number of scenarios')
plt.ylabel('Difference to myopic model objective value (%)')
plt.xticks(rotation=0)  # Rotation des étiquettes de l'axe x si nécessaire
plt.tight_layout()
# plt.show()
fig.set_size_inches(7, 4)

# Échapper les underscores dans les labels et le titre
current_xticklabels = [label.get_text().replace('_', r'\_') for label in plt.gca().get_xticklabels()]
plt.gca().set_xticklabels(current_xticklabels)
plt.ylabel(plt.gca().get_ylabel().replace('_', r'\_'))
plt.xlabel(plt.gca().get_xlabel().replace('_', r'\_'))
if plt.gca().get_title():
    plt.title(plt.gca().get_title().replace('_', r'\_'))

# Échapper les underscores dans la légende
legend = plt.gca().get_legend()
if legend:
    title = legend.get_title()
    if title:
        title.set_text(title.get_text().replace('_', r'\_'))
    for text in legend.get_texts():
        if hasattr(text, 'set_text'):
            text.set_text(text.get_text().replace('_', r'\_'))

plt.savefig('boxplot_aggregation_criterion.pgf')

# Plot machine occupancies evolution.
# TODO : call a function for this.
# machine_occupancies_df = kpis_df[['machine_id_0_load', 'machine_id_1_load', 'machine_id_2_load',
#                                   'machine_id_3_load', 'machine_id_4_load', 'machine_id_5_load',
#                                   'machine_id_6_load']]
#
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=machine_occupancies_df)
# plt.title(f"Evolution of machine occupancies through the simulated scheduling horizon.")
# plt.ylabel("Occupancy (%)")
# plt.xlabel("Day (index)")
# plt.savefig(f"{results_directory_path}/occupancies.png")
#
# plt.figure(figsize=(12, 8))
# sns.lineplot(x=kpis_df['day'], y=kpis_df['sum_qi_treatment_dates'])
# plt.title(f"Evolution of the sum of treatment delays through the simulated scheduling horizon.")
# plt.ylabel("Delays in business days")
# plt.xlabel("Day")
# plt.savefig(f"{results_directory_path}/treatment_delays.png")
#
# plt.figure(figsize=(12, 8))
# sns.lineplot(x=kpis_df['day'], y=kpis_df['sum_qi_count_violated'])
# plt.title(f"Evolution of the sum of periods violated through the simulated scheduling horizon.")
# plt.ylabel("Days of violation")
# plt.xlabel("Day")
# plt.savefig(f"{results_directory_path}/count_violated.png")