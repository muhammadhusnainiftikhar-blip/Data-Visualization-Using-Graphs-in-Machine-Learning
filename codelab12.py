# Lab 12: Data Visualization Using Graphs in Machine Learning

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

# Step 2: Upload Dataset
uploaded = files.upload()

# Step 3: Load Dataset
df = pd.read_csv(list(uploaded.keys())[0])

# Step 4: Show Dataset
print("Dataset Preview:")
print(df.head())

# Step 5: Handle Missing Values (if any)
df = df.dropna()

print("\nCleaned Dataset Info:")
print(df.info())

# ---------------------------------------------------
# ASSUMPTION:
# Dataset should have numeric columns.
# If not, we convert where possible.
# ---------------------------------------------------

# Step 6: Select Numeric Columns
numeric_df = df.select_dtypes(include=[np.number])

print("\nNumeric Columns:")
print(numeric_df.head())

# If dataset is small or unclear, fallback example columns
cols = numeric_df.columns

# If dataset has at least 2 columns, use first two for scatter/line
if len(cols) >= 2:
    x = numeric_df[cols[0]]
    y = numeric_df[cols[1]]
else:
    x = np.arange(len(df))
    y = numeric_df[cols[0]] if len(cols) > 0 else np.arange(len(df))

# ---------------------------------------------------
# 1. LINE GRAPH
# ---------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(x, y, color='blue')
plt.title("Line Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 2. BAR CHART
# ---------------------------------------------------
plt.figure(figsize=(6,4))
plt.bar(x[:10], y[:10], color='green')
plt.title("Bar Chart (First 10 Values)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ---------------------------------------------------
# 3. SCATTER PLOT
# ---------------------------------------------------
plt.figure(figsize=(6,4))
plt.scatter(x, y, color='red')
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 4. HISTOGRAM
# ---------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(y, bins=10, color='purple', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------
# 5. BOX PLOT
# ---------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=y)
plt.title("Box Plot")
plt.show()

print("\nAll graphs generated successfully!")