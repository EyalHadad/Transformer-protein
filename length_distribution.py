import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Load the dataset
df = pd.read_csv('test_dataset.csv')

# Calculate the length of each sequence
df['length'] = df['sequence'].apply(len)

# Determine the maximum sequence length
max_length = df['length'].max()

# Plot the length distribution
plt.figure(figsize=(12, 6))

# Calculate the percentage for each bin
bins = 30
counts, bin_edges = np.histogram(df['length'], bins=bins, range=(0, max_length))
bin_width = (bin_edges[1] - bin_edges[0])
percentages = counts / counts.sum() * 100

# Create a new DataFrame for plotting
plot_df = pd.DataFrame({
    'length': bin_edges[:-1] + bin_width / 2,
    'percentage': percentages
})

# Plot
sns.histplot(data=df, x='length', hue='label', bins=bins, multiple='dodge', stat='percent', common_norm=False)
plt.title('Length Distribution of Protein Sequences by Type')
plt.xlabel('Sequence Length')
plt.ylabel('Percentage of Data')
plt.xlim(0, max_length)
plt.legend(title='Label')
plt.show()
