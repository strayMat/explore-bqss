# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from explore_bqss.data import fetch_bqss_data

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")

# %%
data_dict = fetch_bqss_data()
valeur_df = data_dict["valeur"]
finess_df = data_dict["finess"]
metadata = data_dict["metadata"]
# %%
# Basic dataset exploration

print("=== DATASET OVERVIEW ===")
print(f"Number of datasets loaded: {len(data_dict)}")
print(f"Dataset names: {list(data_dict.keys())}")
print()

# %%
# Explore valeur_df dataset
print("=== VALEUR DATASET ===")
print(f"Shape: {valeur_df.shape}")
print(f"Columns: {list(valeur_df.columns)}")
print(f"Data types:\n{valeur_df.dtypes}")
print(f"Memory usage: {valeur_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

print("First few rows:")
print(valeur_df.head())
print()

print("Basic statistics:")
print(valeur_df.describe(include="all"))
print()

print("Missing values:")
print(valeur_df.isnull().sum())
print()

# %%
# Explore finess_df dataset (Polars DataFrame)
print("=== FINESS DATASET ===")
print(f"Shape: {finess_df.shape}")
print(f"Columns: {list(finess_df.columns)}")
print(f"Data types:\n{finess_df.dtypes}")
# Polars doesn't have memory_usage() method, so we'll estimate
print(f"Estimated memory usage: {finess_df.estimated_size() / 1024**2:.2f} MB")
print()

print("First few rows:")
print(finess_df.head())
print()

print("Basic statistics:")
print(finess_df.describe())
print()

print("Missing values:")
print(finess_df.null_count())
print()

# %%
# Explore metadata dataset
print("=== METADATA DATASET ===")
print(f"Shape: {metadata.shape}")
print(f"Columns: {list(metadata.columns)}")
print(f"Data types:\n{metadata.dtypes}")
print(f"Memory usage: {metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

print("First few rows:")
print(metadata.head())
print()

print("Basic statistics:")
print(metadata.describe(include="all"))
print()

print("Missing values:")
print(metadata.isnull().sum())
print()

# %%
# Visualizations and deeper insights
print("=== VISUALIZATIONS AND INSIGHTS ===")

# Create a figure with subplots for basic visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Dataset Overview Visualizations", fontsize=16)

# 1. Dataset sizes comparison
dataset_sizes = [len(valeur_df), len(finess_df), len(metadata)]
dataset_names = ["Valeur", "Finess", "Metadata"]
axes[0, 0].bar(dataset_names, dataset_sizes, color=["skyblue", "lightcoral", "lightgreen"])
axes[0, 0].set_title("Dataset Sizes (Number of Rows)")
axes[0, 0].set_ylabel("Number of Rows")
for i, v in enumerate(dataset_sizes):
    axes[0, 0].text(i, v + max(dataset_sizes) * 0.01, str(v), ha="center", va="bottom")

# 2. Missing values comparison
missing_data = []
# Handle mixed DataFrame types for missing values calculation
missing_data = []
# Valeur (pandas)
missing_pct = (valeur_df.isnull().sum().sum() / (valeur_df.shape[0] * valeur_df.shape[1])) * 100
missing_data.append(missing_pct)
# Finess (polars) - simplified calculation
null_counts = finess_df.null_count()
total_nulls = sum(null_counts.row(0))
total_cells = finess_df.shape[0] * finess_df.shape[1]
missing_pct = (total_nulls / total_cells) * 100
missing_data.append(missing_pct)
# Metadata (pandas)
missing_pct = (metadata.isnull().sum().sum() / (metadata.shape[0] * metadata.shape[1])) * 100
missing_data.append(missing_pct)

axes[0, 1].bar(dataset_names, missing_data, color=["orange", "red", "yellow"])
axes[0, 1].set_title("Missing Values Percentage")
axes[0, 1].set_ylabel("Missing Values (%)")
for i, v in enumerate(missing_data):
    axes[0, 1].text(i, v + max(missing_data) * 0.01, f"{v:.1f}%", ha="center", va="bottom")

# 3. Memory usage comparison
memory_usage = []
# Valeur (pandas)
memory_mb = valeur_df.memory_usage(deep=True).sum() / 1024**2
memory_usage.append(memory_mb)
# Finess (polars)
memory_mb = finess_df.estimated_size() / 1024**2
memory_usage.append(memory_mb)
# Metadata (pandas)
memory_mb = metadata.memory_usage(deep=True).sum() / 1024**2
memory_usage.append(memory_mb)

axes[1, 0].bar(dataset_names, memory_usage, color=["purple", "brown", "pink"])
axes[1, 0].set_title("Memory Usage (MB)")
axes[1, 0].set_ylabel("Memory (MB)")
for i, v in enumerate(memory_usage):
    axes[1, 0].text(i, v + max(memory_usage) * 0.01, f"{v:.1f}", ha="center", va="bottom")

# 4. Number of columns comparison
column_counts = [len(valeur_df.columns), len(finess_df.columns), len(metadata.columns)]
axes[1, 1].bar(dataset_names, column_counts, color=["cyan", "magenta", "lime"])
axes[1, 1].set_title("Number of Columns")
axes[1, 1].set_ylabel("Column Count")
for i, v in enumerate(column_counts):
    axes[1, 1].text(i, v + max(column_counts) * 0.01, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()

# %%
# Data type distribution analysis
print("=== DATA TYPES DISTRIBUTION ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Data Types Distribution Across Datasets", fontsize=16)

# Handle data type visualization for mixed DataFrame types
dataset_info = [("Valeur", valeur_df, "pandas"), ("Finess", finess_df, "polars"), ("Metadata", metadata, "pandas")]
for idx, (name, df, df_type) in enumerate(dataset_info):
    if df_type == "pandas":
        dtype_counts = df.dtypes.value_counts()
        axes[idx].pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.1f%%", startangle=90)
    else:  # polars
        # For polars, we'll count the string representation of dtypes
        dtype_strs = [str(dtype) for dtype in df.dtypes]
        dtype_counts = pd.Series(dtype_strs).value_counts()
        axes[idx].pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.1f%%", startangle=90)
    axes[idx].set_title(f"{name} Dataset\nData Types")

plt.tight_layout()
plt.show()

# %%
# Summary insights
print("=== KEY INSIGHTS ===")
print("📊 Dataset Summary:")
print(f"   • Total records across all datasets: {sum([len(valeur_df), len(finess_df), len(metadata)]):,}")
print(
    f"   • Largest dataset: {max(zip(dataset_names, dataset_sizes), key=lambda x: x[1])[0]} ({max(dataset_sizes):,} rows)"
)
print(f"   • Total memory usage: {sum(memory_usage):.2f} MB")
print()

print("🔍 Data Quality:")
overall_missing = sum(missing_data) / len(missing_data)
print(f"   • Average missing values: {overall_missing:.2f}%")
if overall_missing < 5:
    print("   ✅ Good data quality - low missing values")
elif overall_missing < 15:
    print("   ⚠️  Moderate data quality - some missing values")
else:
    print("   ❌ Poor data quality - high missing values")

print()
print("💡 Next Steps for Analysis:")
print("   • Examine relationships between datasets")
print("   • Identify key variables for analysis")
print("   • Check for data consistency across datasets")
print("   • Perform domain-specific analysis based on BQSS context")

# %%
# Advanced exploration - data quality and patterns
print("=== DATA QUALITY ANALYSIS ===")

# Check for duplicates
print("Duplicate rows:")
# Handle duplicates check for mixed DataFrame types
datasets = [("valeur", valeur_df, "pandas"), ("finess", finess_df, "polars"), ("metadata", metadata, "pandas")]
for name, df, df_type in datasets:
    if df_type == "pandas":
        duplicates = df.duplicated().sum()
    else:  # polars
        duplicates = df.is_duplicated().sum()
    print(f"  {name}: {duplicates} duplicate rows ({duplicates / len(df) * 100:.2f}%)")

print()

# Check unique values in categorical columns
print("Unique values in categorical columns (showing first 10):")
for name, df, df_type in datasets:
    print(f"\n{name.upper()} dataset:")
    if df_type == "pandas":
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            unique_count = df[col].nunique()
            unique_values = df[col].unique()[:10]  # Show first 10 unique values
            print(f"  {col}: {unique_count} unique values")
            print(f"    Sample values: {list(unique_values)}")
    else:  # polars
        string_cols = [col for col, dtype in zip(df.columns, df.dtypes) if str(dtype).startswith("String")]
        for col in string_cols[:5]:  # Limit to first 5 string columns
            unique_count = df[col].n_unique()
            unique_values = df[col].unique().limit(10).to_list()
            print(f"  {col}: {unique_count} unique values")
            print(f"    Sample values: {unique_values}")

# %%
