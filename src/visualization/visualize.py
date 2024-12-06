import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_processed = pd.read_csv('diabetes_processed.csv')

binary_var = 'diabetes' 

for col in diabetes_processed.columns:
    if col == binary_var:
        continue  # Skip the binary variable itself
    
    # Check if the column is categorical
    if diabetes_processed[col].dtype == 'object' or diabetes_processed[col].nunique() <= 10:
        # Create a pivot table to prepare data for stacked bar plot
        stacked_data = diabetes_processed.groupby([col, binary_var]).size().unstack(fill_value=0)

        # Plot stacked bar plot
        ax = stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#79a3b1', '#ef9b20'])
        plt.title(f'Stacked Bar Plot of {col} by {binary_var}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.legend(title=binary_var, loc='upper right')

        # Annotate the bars with counts
        for i, bars in enumerate(ax.containers):  # Iterate through the bar segments
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,  # x-coordinate (center of the bar)
                        bar.get_y() + height / 2,  # y-coordinate (center of the bar segment)
                        f'{int(height)}',  # Annotation text (count)
                        ha='center', va='center', fontsize=10, color='black'
                    )
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping {col} (not categorical)")