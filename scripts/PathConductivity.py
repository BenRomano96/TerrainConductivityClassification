#%%
import pandas as pd
from IPython.core.display import display, HTML

# --------------------------------------------------------------------
# Merge coordinate data with terrain conductivity predictions into
# one final result table, then export as HTML and CSV.

# Required input files:
#   - CoordinatesPath.csv': generated by TerrainPath.py
#   - ConductivityTable.csv : generated by TerrainConductivityTable.py
# --------------------------------------------------------------------

# Load input CSVs
coor_df = pd.read_csv('csv/CoordinatesPath.csv', index_col=0)        
cond_df = pd.read_csv('csv/ConductivityTable.csv', index_col=0)      

# Rename 'Terrain' to 'Type' for clarity
coor_df.rename(columns={'Terrain': 'Type'}, inplace=True)

# Keep only land points
coor_df = coor_df[coor_df['Type'] != 'Water']

# Sort conductivity data by kilometer marker
cond_df_sort = cond_df.sort_values(by='Km', ignore_index=True)

# Merge both DataFrames side by side
final_df = pd.concat([coor_df.reset_index(drop=True),
                      cond_df_sort.reset_index(drop=True)], axis=1)

# Display table with image thumbnails
def path_to_image_html(path):
    return f'<img src="{path}" width="60" >'

pd.set_option('display.max_colwidth', None)
format_dict = {'image': path_to_image_html}

# Show HTML table and save it
display(HTML(final_df.to_html(escape=False, formatters=format_dict, index=False)))
final_df.to_html('test_html2.html', escape=False, formatters=format_dict, index=False)

# Save final merged table as CSV
final_df.to_csv('condactivityPath.csv', index=False)

# Optional: Load and print average conductivity from a specific range
folder = 'csv/conductivity_path.csv'
cond_path = pd.read_csv(folder, index_col=0)
print(cond_path['Conductivity'].iloc[30:50].mean(), "[mS/m]")
