import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.stats import linregress
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
import calendar
from sklearn.cluster import KMeans
from pandas.plotting import andrews_curves, parallel_coordinates
import squarify  # You may need to run: pip install squarify
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

# Generate sample datasets
np.random.seed(42)
n = 100

# For correlation plots
df_corr = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n) * 2 + 1,
    'z': np.random.randn(n) * 0.5,
    'category': np.random.choice(['A', 'B', 'C'], n),
    'size': np.random.randint(20, 200, n)
})
df_corr['y_corr'] = df_corr['x'] * 2 + np.random.randn(n) * 0.5

# For time series
dates_ts = pd.date_range('2023-01-01', periods=365, freq='D')
df_ts = pd.DataFrame({
    'date': dates_ts,
    'value': np.cumsum(np.random.randn(365)) + 100,
    'value2': np.cumsum(np.random.randn(365)) + 50
})

# For categorical data
df_cat = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value': [23, 45, 56, 78, 32],
    'value2': [34, 56, 23, 67, 89]
})

print("=" * 80)
print("DATA SCIENCE VISUALIZATION CODE COLLECTION")
print("=" * 80)

# ============================================================================
# 1. CORRELATION
# ============================================================================
print("\n### 1. CORRELATION PLOTS (Running) ###\n")

# 1.1 Scatter Plot
print("# 1.1 Basic Scatter Plot")
plt.figure(figsize=(8, 6))
plt.scatter(df_corr['x'], df_corr['y'], alpha=0.6, s=50)
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.title('1.1 Scatter Plot')
plt.grid(True, alpha=0.3)
plt.show()

# 1.2 Bubble Plot with Encircling
print("\n# 1.2 Bubble Plot with Encircling")
plt.figure(figsize=(10, 6))
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}

for cat in df_corr['category'].unique():
    subset = df_corr[df_corr['category'] == cat]
    plt.scatter(subset['x'], subset['y'],
                s=subset['size'],
                c=colors[cat],
                alpha=0.6,
                label=cat)

    # Encircling with convex hull
    if len(subset) > 2:
        points = subset[['x', 'y']].values
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1],
                     c=colors[cat], alpha=0.3, linestyle='--')

plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.title('1.2 Bubble Plot with Encircling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 1.3 Scatter Plot with Line of Best Fit
print("\n# 1.3 Scatter Plot with Line of Best Fit")
plt.figure(figsize=(8, 6))
plt.scatter(df_corr['x'], df_corr['y_corr'], alpha=0.6, s=50, label='Data')

# Line of best fit
slope, intercept, r_value, p_value, std_err = linregress(df_corr['x'], df_corr['y_corr'])
line = slope * df_corr['x'] + intercept
plt.plot(df_corr['x'], line, 'r-', linewidth=2,
         label=f'Best Fit (R²={r_value ** 2:.3f})')

plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.title('1.3 Scatter Plot with Line of Best Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 1.4 Jittering with Stripplot
print("\n# 1.4 Jittering with Stripplot")
plt.figure(figsize=(8, 6))
sns.stripplot(data=df_corr, x='category', y='y', jitter=True, alpha=0.6, s=8)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('1.4 Strip Plot with Jittering')
plt.show()

# 1.5 Counts Plot
print("\n# 1.5 Counts Plot")
plt.figure(figsize=(8, 6))
sns.countplot(data=df_corr, x='category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('1.5 Counts Plot')
plt.show()

# 1.6 Marginal Histogram
print("\n# 1.6 Marginal Histogram")
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 4)

ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

ax_main.scatter(df_corr['x'], df_corr['y'], alpha=0.6)
ax_main.set(xlabel='X', ylabel='Y')

ax_xDist.hist(df_corr['x'], bins=30, color='blue', alpha=0.7)
ax_xDist.set(ylabel='Frequency')
ax_xDist.tick_params(labelbottom=False)

ax_yDist.hist(df_corr['y'], bins=30, orientation='horizontal', color='blue', alpha=0.7)
ax_yDist.set(xlabel='Frequency')
ax_yDist.tick_params(labelleft=False)

plt.suptitle('1.6 Scatter Plot with Marginal Histograms')
plt.tight_layout()
plt.show()

# 1.7 Marginal Boxplot
print("\n# 1.7 Marginal Boxplot")
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 4)

ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_xBox = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_yBox = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

ax_main.scatter(df_corr['x'], df_corr['y'], alpha=0.6)
ax_main.set(xlabel='X', ylabel='Y')

ax_xBox.boxplot(df_corr['x'], vert=False, widths=0.7)
ax_xBox.tick_params(labelbottom=False, labelleft=False)

ax_yBox.boxplot(df_corr['y'], widths=0.7)
ax_yBox.tick_params(labelleft=False, labelbottom=False)

plt.suptitle('1.7 Scatter Plot with Marginal Boxplots')
plt.tight_layout()
plt.show()

# 1.8 Correlogram
print("\n# 1.8 Correlogram")
df_numeric = df_corr.select_dtypes(include=[np.number])
corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('1.8 Correlogram (Correlation Matrix)')
plt.tight_layout()
plt.show()

# 1.9 Pairwise Plot
print("\n# 1.9 Pairwise Plot")
sns.pairplot(df_corr[['x', 'y', 'z', 'category']], hue='category',
             diag_kind='kde', height=2.5)
plt.suptitle('1.9 Pairwise Plot', y=1.02)
plt.show()

# ============================================================================
# 2. DEVIATION
# ============================================================================
print("\n### 2. DEVIATION PLOTS (Running) ###\n")

# 2.1 Diverging Bars
print("# 2.1 Diverging Bars")
df_div = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value': [10, -5, 15, -8, 12]
})

plt.figure(figsize=(8, 6))
colors_div = ['green' if x > 0 else 'red' for x in df_div['value']]
plt.barh(df_div['category'], df_div['value'], color=colors_div, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('2.1 Diverging Bar Chart')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 2.2 Diverging Texts
print("\n# 2.2 Diverging Texts")
plt.figure(figsize=(10, 6))
colors_div = ['green' if x > 0 else 'red' for x in df_div['value']]
plt.barh(df_div['category'], df_div['value'], color=colors_div, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

for i, (cat, val) in enumerate(zip(df_div['category'], df_div['value'])):
    plt.text(val + (0.5 if val > 0 else -0.5), i, str(val),
             va='center', ha='left' if val > 0 else 'right')

plt.xlabel('Value')
plt.ylabel('Category')
plt.title('2.2 Diverging Bar Chart with Text Annotations')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 2.3 Diverging Dot Plot
print("\n# 2.3 Diverging Dot Plot")
plt.figure(figsize=(8, 6))
colors_div = ['green' if x > 0 else 'red' for x in df_div['value']]
plt.scatter(df_div['value'], df_div['category'], s=200, c=colors_div, alpha=0.7, edgecolors='black')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('2.3 Diverging Dot Plot')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 2.4 Diverging Lollipop Chart
print("\n# 2.4 Diverging Lollipop Chart with Markers")
plt.figure(figsize=(8, 6))
colors_div = ['green' if x > 0 else 'red' for x in df_div['value']]

for i, (cat, val, col) in enumerate(zip(df_div['category'], df_div['value'], colors_div)):
    plt.plot([0, val], [i, i], color=col, linewidth=2, alpha=0.7)
    plt.scatter(val, i, s=200, color=col, alpha=0.7, edgecolors='black', zorder=3)

plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.yticks(range(len(df_div)), df_div['category'])
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('2.4 Diverging Lollipop Chart')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 2.5 Area Chart
print("\n# 2.5 Area Chart")
x = np.arange(0, 10, 0.1)
y_baseline = np.sin(x)
y_data = y_baseline + np.random.randn(len(x)) * 0.1

plt.figure(figsize=(10, 6))
plt.fill_between(x, y_baseline, y_data, alpha=0.5, label='Deviation')
plt.plot(x, y_baseline, 'r-', linewidth=2, label='Baseline')
plt.plot(x, y_data, 'b-', linewidth=1, alpha=0.7, label='Actual')
plt.xlabel('X')
plt.ylabel('Value')
plt.title('2.5 Area Chart Showing Deviation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 3. RANKING
# ============================================================================
print("\n### 3. RANKING PLOTS (Running) ###\n")

# 3.1 Ordered Bar Chart
print("# 3.1 Ordered Bar Chart")
df_sorted = df_cat.sort_values('value', ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(df_sorted['category'], df_sorted['value'], color='steelblue', alpha=0.8)
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('3.1 Ordered Bar Chart')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 3.2 Lollipop Chart
print("\n# 3.2 Lollipop Chart")
df_sorted = df_cat.sort_values('value', ascending=True)

plt.figure(figsize=(8, 6))
plt.hlines(y=df_sorted['category'], xmin=0, xmax=df_sorted['value'],
           color='steelblue', alpha=0.7, linewidth=3)
plt.scatter(df_sorted['value'], df_sorted['category'], s=200,
            color='steelblue', alpha=0.8, edgecolors='black', zorder=3)
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('3.2 Lollipop Chart')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 3.3 Dot Plot
print("\n# 3.3 Dot Plot")
df_sorted = df_cat.sort_values('value', ascending=True)

plt.figure(figsize=(8, 6))
plt.scatter(df_sorted['value'], df_sorted['category'], s=300,
            color='steelblue', alpha=0.7, edgecolors='black')
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('3.3 Dot Plot')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 3.4 Slope Chart
print("\n# 3.4 Slope Chart")
df_slope = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'value_2022': [23, 45, 56, 78, 32],
    'value_2023': [34, 56, 43, 89, 38]
})

plt.figure(figsize=(10, 8))
for i in range(len(df_slope)):
    plt.plot([0, 1],
             [df_slope['value_2022'].iloc[i], df_slope['value_2023'].iloc[i]],
             'o-', linewidth=2, markersize=8, alpha=0.7)
    plt.text(-0.05, df_slope['value_2022'].iloc[i], df_slope['category'].iloc[i],
             ha='right', va='center', fontsize=10)

plt.xticks([0, 1], ['2022', '2023'])
plt.ylabel('Value')
plt.title('3.4 Slope Chart')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 3.5 Dumbbell Plot
print("\n# 3.5 Dumbbell Plot")
plt.figure(figsize=(10, 6))
for i in range(len(df_slope)):
    plt.plot([df_slope['value_2022'].iloc[i], df_slope['value_2023'].iloc[i]],
             [i, i], 'o-', linewidth=2, markersize=10, alpha=0.7,
             label=df_slope['category'].iloc[i] if i == 0 else "")  # Simple legend

# Re-plot points to show both colors
plt.scatter(df_slope['value_2022'], range(len(df_slope)), color='blue', s=100, label='2022')
plt.scatter(df_slope['value_2023'], range(len(df_slope)), color='orange', s=100, label='2023')

plt.yticks(range(len(df_slope)), df_slope['category'])
plt.xlabel('Value')
plt.ylabel('Category')
plt.title('3.5 Dumbbell Plot (2022 vs 2023)')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# ============================================================================
# 4. DISTRIBUTION
# ============================================================================
print("\n### 4. DISTRIBUTION PLOTS (Running) ###\n")

# 4.1 Histogram for Continuous Variable
print("# 4.1 Histogram for Continuous Variable")
data_dist_cont = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(data_dist_cont, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('4.1 Histogram for Continuous Variable')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 4.2 Histogram for Categorical Variable
print("\n# 4.2 Histogram for Categorical Variable")
categories_dist = ['A', 'B', 'C', 'D', 'E']
counts_dist = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
plt.bar(categories_dist, counts_dist, color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('4.2 Histogram for Categorical Variable (Bar Chart)')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 4.3 Density Plot
print("\n# 4.3 Density Plot")
data_dist_cont = np.random.randn(1000)

plt.figure(figsize=(8, 6))
sns.kdeplot(data_dist_cont, fill=True, linewidth=2)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('4.3 Density Plot')
plt.grid(True, alpha=0.3)
plt.show()

# 4.4 Density Curves with Histogram
print("\n# 4.4 Density Curves with Histogram")
data_dist_cont = np.random.randn(1000)

plt.figure(figsize=(8, 6))
sns.histplot(data_dist_cont, bins=30, kde=True, stat='density',
             alpha=0.5, color='steelblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('4.4 Histogram with Density Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4.5 Joy Plot (Ridgeline Plot)
print("\n# 4.5 Joy Plot (Ridgeline Plot)")
# Note: Joy Plots are easier with libraries like 'joypy', but this is a manual way.
categories_joy = ['A', 'B', 'C', 'D']
data_joy = [np.random.randn(500) + i * 2 for i in range(len(categories_joy))]

fig, axes = plt.subplots(len(categories_joy), 1, figsize=(10, 8), sharex=True, sharey=True)
fig.subplots_adjust(hspace=-0.5)  # Overlap plots

for i, (d, cat) in enumerate(zip(data_joy, categories_joy)):
    ax = axes[i]
    sns.kdeplot(d, fill=True, ax=ax, alpha=0.7)
    ax.set_ylabel(cat, rotation=0, labelpad=20, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel(cat, rotation=0, ha='right', va='center', labelpad=20)

    # Add a white background to obscure lines below
    ax.set_facecolor('white')
    ax.patch.set_alpha(1)

axes[-1].set_xlabel('Value')
axes[-1].spines['bottom'].set_visible(True)
plt.suptitle('4.5 Joy Plot (Ridgeline Plot)', y=0.98)
plt.show()

# 4.6 Distributed Dot Plot
print("\n# 4.6 Distributed Dot Plot")
data_dot = np.random.randn(100)

plt.figure(figsize=(10, 4))
sns.stripplot(x=data_dot, jitter=0.05, alpha=0.7, s=8)
plt.xlabel('Value')
plt.title('4.6 Distributed Dot Plot (Stripplot)')
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 4.7 Box Plot
print("\n# 4.7 Box Plot")
data_box = [np.random.randn(100) + i for i in range(5)]

plt.figure(figsize=(8, 6))
plt.boxplot(data_box, labels=['A', 'B', 'C', 'D', 'E'], patch_artist=True)
plt.ylabel('Value')
plt.xlabel('Category')
plt.title('4.7 Box Plot')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 4.8 Dot + Box Plot
print("\n# 4.8 Dot + Box Plot")
data_dict = {'A': np.random.randn(50),
             'B': np.random.randn(50) + 1,
             'C': np.random.randn(50) - 1}
df_dotbox = pd.DataFrame(data_dict)

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df_dotbox, ax=ax, palette='pastel')
sns.stripplot(data=df_dotbox, ax=ax, color='black', jitter=True, alpha=0.3, s=4)

ax.set_ylabel('Value')
ax.set_title('4.8 Box Plot with Dots')
ax.grid(True, alpha=0.3, axis='y')
plt.show()

# 4.9 Violin Plot
print("\n# 4.9 Violin Plot")
data_violin = [np.random.randn(100) + i for i in range(5)]
df_violin = pd.DataFrame(dict(zip(['A', 'B', 'C', 'D', 'E'], data_violin)))

plt.figure(figsize=(8, 6))
sns.violinplot(data=df_violin, inner='quartile', palette='pastel')
plt.ylabel('Value')
plt.xlabel('Category')
plt.title('4.9 Violin Plot')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 4.10 Population Pyramid
print("\n# 4.10 Population Pyramid")
age_groups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
male = np.array([5, 6, 8, 7, 6, 5, 3, 2])
female = np.array([4, 5, 7, 8, 7, 6, 4, 3])

y_pos = np.arange(len(age_groups))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y_pos, male, align='center', alpha=0.7, label='Male', color='steelblue')
ax.barh(y_pos, -female, align='center', alpha=0.7, label='Female', color='coral')
ax.set_yticks(y_pos)
ax.set_yticklabels(age_groups)
ax.set_xlabel('Population (thousands)')
ax.set_title('4.10 Population Pyramid')
ax.legend()
ax.axvline(x=0, color='black', linewidth=0.8)
# Format x-axis to show positive numbers on both sides
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick)) for tick in ticks])
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# 4.11 Categorical Plots
print("\n# 4.11 Categorical Plots (Various)")
# Using seaborn for categorical plots
df_cat_plot = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C'], 30),
    'value': np.concatenate([np.random.randn(30),
                             np.random.randn(30) + 1,
                             np.random.randn(30) - 1])
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.stripplot(data=df_cat_plot, x='category', y='value', ax=axes[0, 0])
axes[0, 0].set_title('Strip Plot')

sns.swarmplot(data=df_cat_plot, x='category', y='value', ax=axes[0, 1])
axes[0, 1].set_title('Swarm Plot')

sns.boxplot(data=df_cat_plot, x='category', y='value', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot')

sns.violinplot(data=df_cat_plot, x='category', y='value', ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot')

plt.suptitle('4.11 Various Categorical Plots')
plt.tight_layout()
plt.show()

# ============================================================================
# 5. COMPOSITION
# ============================================================================
print("\n### 5. COMPOSITION PLOTS (Running) ###\n")

# 5.1 Waffle Chart
print("# 5.1 Waffle Chart")
# Simple waffle chart implementation
categories_waffle = {'A': 30, 'B': 25, 'C': 20, 'D': 15, 'E': 10}
total_waffle = sum(categories_waffle.values())
if total_waffle != 100:
    print("Warning: Waffle chart values don't sum to 100, scaling.")
    # Simple scaling
    categories_waffle = {k: int(v * 100 / total_waffle) for k, v in categories_waffle.items()}
    # Adjust last item to ensure sum is 100
    diff = 100 - sum(categories_waffle.values())
    categories_waffle[list(categories_waffle.keys())[-1]] += diff

fig, ax = plt.subplots(figsize=(10, 6))
colors_waffle = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
color_dict = dict(zip(categories_waffle.keys(), colors_waffle))

values_cumsum = np.cumsum(list(categories_waffle.values()))
previous = 0
for i, (cat, val) in enumerate(categories_waffle.items()):
    for j in range(previous, values_cumsum[i]):
        row = 9 - (j // 10)  # Start from top-left
        col = j % 10
        ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                   facecolor=color_dict[cat],
                                   edgecolor='white', linewidth=2))
    previous = values_cumsum[i]

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Legend
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_dict[cat],
                                 label=f'{cat}: {val}%')
                   for cat, val in categories_waffle.items()]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.title('5.1 Waffle Chart (Each square = 1%)', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 5.2 Pie Chart
print("\n# 5.2 Pie Chart")
categories_pie = ['A', 'B', 'C', 'D', 'E']
sizes_pie = [30, 25, 20, 15, 10]
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
explode_pie = (0.1, 0, 0, 0, 0)

plt.figure(figsize=(8, 8))
plt.pie(sizes_pie, explode=explode_pie, labels=categories_pie, colors=colors_pie,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('5.2 Pie Chart')
plt.axis('equal')
plt.show()

# 5.3 Treemap
print("\n# 5.3 Treemap")
categories_tree = ['A', 'B', 'C', 'D', 'E']
sizes_tree = [30, 25, 20, 15, 10]
colors_tree = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
labels_tree = [f"{cat}\n({sz}%)" for cat, sz in zip(categories_tree, sizes_tree)]

plt.figure(figsize=(10, 8))
squarify.plot(sizes=sizes_tree, label=labels_tree, color=colors_tree, alpha=0.8,
              text_kwargs={'fontsize': 12, 'weight': 'bold'})
plt.title('5.3 Treemap', fontsize=16)
plt.axis('off')
plt.show()

# 5.4 Bar Chart (Stacked)
print("\n# 5.4 Stacked Bar Chart")
categories_bar = ['Q1', 'Q2', 'Q3', 'Q4']
product_a = [23, 45, 56, 78]
product_b = [34, 56, 23, 67]
product_c = [12, 34, 45, 23]

x_bar = np.arange(len(categories_bar))
width_bar = 0.6

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_bar, product_a, width_bar, label='Product A', color='#FF6B6B', alpha=0.8)
ax.bar(x_bar, product_b, width_bar, bottom=product_a, label='Product B',
       color='#4ECDC4', alpha=0.8)
ax.bar(x_bar, product_c, width_bar,
       bottom=np.array(product_a) + np.array(product_b),
       label='Product C', color='#45B7D1', alpha=0.8)

ax.set_ylabel('Sales')
ax.set_xlabel('Quarter')
ax.set_title('5.4 Stacked Bar Chart (Composition)')
ax.set_xticks(x_bar)
ax.set_xticklabels(categories_bar)
ax.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# ============================================================================
# 6. CHANGE (TIME SERIES)
# ============================================================================
print("\n### 6. CHANGE (TIME SERIES) PLOTS (Running) ###\n")

# 6.1 Time Series Plot
print("# 6.1 Time Series Plot")
dates_ts = pd.date_range('2023-01-01', periods=365, freq='D')
values_ts = np.cumsum(np.random.randn(365)) + 100

plt.figure(figsize=(12, 6))
plt.plot(dates_ts, values_ts, linewidth=2, color='steelblue')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('6.1 Time Series Plot')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.2 Time Series with Peaks and Troughs
print("\n# 6.2 Time Series with Peaks and Troughs")
peaks, _ = find_peaks(values_ts, distance=20)
troughs, _ = find_peaks(-values_ts, distance=20)

plt.figure(figsize=(12, 6))
plt.plot(dates_ts, values_ts, linewidth=2, color='steelblue', label='Time Series')
plt.scatter(dates_ts[peaks], values_ts[peaks], color='green', s=100,
            label='Peaks', zorder=5, marker='^')
plt.scatter(dates_ts[troughs], values_ts[troughs], color='red', s=100,
            label='Troughs', zorder=5, marker='v')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('6.2 Time Series with Peaks and Troughs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.3 Autocorrelation Plot
print("\n# 6.3 Annotated Autocorrelation Plot")
values_acf = np.cumsum(np.random.randn(100))

fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(values_acf, lags=40, ax=ax)
plt.title('6.3 Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6.4 Cross Correlation Plot
print("\n# 6.4 Cross Correlation Plot")
series1_cc = np.cumsum(np.random.randn(100))
series2_cc = np.roll(series1_cc, 5) + np.random.randn(100) * 0.5

plt.figure(figsize=(12, 6))
plt.xcorr(series1_cc, series2_cc, maxlags=50, usevlines=True, normed=True)
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('6.4 Cross Correlation Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6.5 Time Series Decomposition
print("\n# 6.5 Time Series Decomposition Plot")
# Create time series with trend, seasonality, and noise
dates_decomp = pd.date_range('2020-01-01', periods=365 * 2, freq='D')
trend_decomp = np.linspace(0, 100, len(dates_decomp))
seasonal_decomp = 10 * np.sin(np.arange(len(dates_decomp)) * 2 * np.pi / 30.5)  # Monthly
noise_decomp = np.random.randn(len(dates_decomp)) * 2
ts_data_decomp = trend_decomp + seasonal_decomp + noise_decomp

ts_df_decomp = pd.DataFrame({'date': dates_decomp, 'value': ts_data_decomp})
ts_df_decomp.set_index('date', inplace=True)

# Using a shorter period for clearer decomposition
decomposition = seasonal_decompose(ts_df_decomp['value'], model='additive', period=30)

fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.suptitle('6.5 Time Series Decomposition Plot', y=1.02)
plt.tight_layout()
plt.show()

# 6.6 Multiple Time Series
print("\n# 6.6 Multiple Time Series")
dates_multi = pd.date_range('2023-01-01', periods=365, freq='D')
series1_multi = np.cumsum(np.random.randn(365)) + 100
series2_multi = np.cumsum(np.random.randn(365)) + 80
series3_multi = np.cumsum(np.random.randn(365)) + 120

plt.figure(figsize=(12, 6))
plt.plot(dates_multi, series1_multi, label='Series 1', linewidth=2)
plt.plot(dates_multi, series2_multi, label='Series 2', linewidth=2)
plt.plot(dates_multi, series3_multi, label='Series 3', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('6.6 Multiple Time Series')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.7 Secondary Y-axis
print("\n# 6.7 Plotting with Different Scales (Secondary Y-axis)")
dates_sec = pd.date_range('2023-01-01', periods=365, freq='D')
series1_sec = np.cumsum(np.random.randn(365)) + 100
series2_sec = np.cumsum(np.random.randn(365)) * 1000 + 50000

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Series 1', color=color1)
ax1.plot(dates_sec, series1_sec, color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, which='major', linestyle='--')

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Series 2', color=color2)
ax2.plot(dates_sec, series2_sec, color=color2, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('6.7 Time Series with Secondary Y-axis')
plt.xticks(rotation=45)
fig.tight_layout()
plt.show()

# 6.8 Time Series with Error Bands
print("\n# 6.8 Time Series with Error Bands")
dates_err = pd.date_range('2023-01-01', periods=365, freq='D')
mean_values_err = np.cumsum(np.random.randn(365)) + 100
std_values_err = np.abs(np.random.randn(365) * 2 + 5)  # Generate some error values

plt.figure(figsize=(12, 6))
plt.plot(dates_err, mean_values_err, linewidth=2, color='steelblue', label='Mean')
plt.fill_between(dates_err,
                 mean_values_err - std_values_err,
                 mean_values_err + std_values_err,
                 alpha=0.3, color='steelblue', label='Error Band (± 1 Std Dev)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('6.8 Time Series with Error Bands')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.9 Stacked Area Chart
print("\n# 6.9 Stacked Area Chart")
dates_stack = pd.date_range('2023-01-01', periods=100, freq='D')
y1_stack = np.abs(np.random.randn(100).cumsum()) + 10
y2_stack = np.abs(np.random.randn(100).cumsum()) + 10
y3_stack = np.abs(np.random.randn(100).cumsum()) + 10

plt.figure(figsize=(12, 6))
plt.stackplot(dates_stack, y1_stack, y2_stack, y3_stack,
              labels=['Product A', 'Product B', 'Product C'],
              alpha=0.8, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xlabel('Date')
plt.ylabel('Total Value')
plt.title('6.9 Stacked Area Chart')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.10 Unstacked Area Chart (from syllabus: "Area Chart Unstacked")
print("\n# 6.10 Unstacked Area Chart")
dates_unstack = pd.date_range('2023-01-01', periods=365, freq='D')
y1_unstack = np.cumsum(np.random.randn(365)) + 100
y2_unstack = np.cumsum(np.random.randn(365)) + 90
y3_unstack = np.cumsum(np.random.randn(365)) + 110

plt.figure(figsize=(12, 6))
# Plot with transparency to see overlaps
plt.fill_between(dates_unstack, y1_unstack, alpha=0.5, label='Series 1')
plt.fill_between(dates_unstack, y2_unstack, alpha=0.5, label='Series 2')
plt.fill_between(dates_unstack, y3_unstack, alpha=0.5, label='Series 3')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('6.10 Unstacked Area Chart')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 6.12 Seasonal Plot
print("\n# 6.12 Seasonal Plot")
# Generate 3 years of monthly data
dates_seasonal = pd.date_range('2022-01-01', '2024-12-31', freq='MS')
values_seasonal = 50 + 20 * np.sin(np.arange(len(dates_seasonal)) * 2 * np.pi / 12) + np.random.randn(
    len(dates_seasonal)) * 5

df_seasonal = pd.DataFrame({'date': dates_seasonal, 'value': values_seasonal})
df_seasonal['year'] = df_seasonal['date'].dt.year
df_seasonal['month'] = df_seasonal['date'].dt.month

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_seasonal, x='month', y='value', hue='year',
             marker='o', linewidth=2, palette='viridis')

plt.xlabel('Month')
plt.ylabel('Value')
plt.title('6.12 Seasonal Plot')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# 7. GROUPS
# ============================================================================
print("\n### 7. GROUPS PLOTS (Running) ###\n")

# 7.1 Dendrogram
print("# 7.1 Dendrogram")
# Generate sample data
X_dendro = np.random.randn(50, 4)
# Scale data for better clustering
X_dendro_scaled = StandardScaler().fit_transform(X_dendro)

# Perform hierarchical clustering
Z = linkage(X_dendro_scaled, method='ward')

plt.figure(figsize=(12, 7))
dendrogram(Z, leaf_font_size=10, leaf_rotation=90)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('7.1 Dendrogram (Hierarchical Clustering)')
plt.tight_layout()
plt.show()

# 7.2 Cluster Plot
print("\n# 7.2 Cluster Plot")
# Generate sample data
np.random.seed(42)
X_cluster = np.random.randn(300, 2)
X_cluster[:100] += [2, 2]
X_cluster[100:200] += [-2, -2]
X_cluster[200:] += [2, -2]

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_cluster = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels_cluster, cmap='viridis',
                      s=50, alpha=0.6, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='red', s=300, alpha=0.8, marker='X',
            edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('7.2 Cluster Plot (K-Means)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7.3 Andrews Curve
print("\n# 7.3 Andrews Curve")
# Generate sample data
df_andrews = pd.DataFrame({
    'feature1': np.random.randn(150),
    'feature2': np.random.randn(150),
    'feature3': np.random.randn(150),
    'feature4': np.random.randn(150),
    'class': np.repeat(['A', 'B', 'C'], 50)
})
# Scale data first
df_andrews.iloc[:, 0:4] = StandardScaler().fit_transform(df_andrews.iloc[:, 0:4])

plt.figure(figsize=(12, 6))
andrews_curves(df_andrews, 'class', alpha=0.4)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('7.3 Andrews Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7.4 Parallel Coordinates
print("\n# 7.4 Parallel Coordinates")
# Generate sample data
df_parallel = pd.DataFrame({
    'feature1': np.random.randn(150),
    'feature2': np.random.randn(150),
    'feature3': np.random.randn(150),
    'feature4': np.random.randn(150),
    'class': np.repeat(['A', 'B', 'C'], 50)
})
# Scale data first
df_parallel.iloc[:, 0:4] = StandardScaler().fit_transform(df_parallel.iloc[:, 0:4])

plt.figure(figsize=(12, 6))
parallel_coordinates(df_parallel, 'class', alpha=0.4)
plt.ylabel('Value (Standardized)')
plt.title('7.4 Parallel Coordinates Plot')
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ============================================================================
# COMPLETE EXAMPLE: COMPREHENSIVE DASHBOARD
# ============================================================================
print("\n### BONUS: COMPREHENSIVE DASHBOARD (Running) ###\n")

# Create a comprehensive dashboard with multiple visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# 1. Time Series
ax1 = fig.add_subplot(gs[0, :])
dates_dash = pd.date_range('2023-01-01', periods=365, freq='D')
values_dash = np.cumsum(np.random.randn(365)) + 100
ax1.plot(dates_dash, values_dash, linewidth=2, color='steelblue')
ax1.set_title('Time Series', fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

# 2. Distribution
ax2 = fig.add_subplot(gs[1, 0])
data_dash = np.random.randn(1000)
sns.histplot(data_dash, bins=30, color='coral', alpha=0.7, ax=ax2, kde=True)
ax2.set_title('Distribution', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Correlation
ax3 = fig.add_subplot(gs[1, 1])
x_dash = np.random.randn(100)
y_dash = x_dash * 2 + np.random.randn(100) * 0.5
sns.regplot(x=x_dash, y=y_dash, ax=ax3, color='green',
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red'})
ax3.set_title('Correlation', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Pie Chart
ax4 = fig.add_subplot(gs[1, 2])
sizes_dash = [30, 25, 20, 15, 10]
colors_pie_dash = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
ax4.pie(sizes_dash, labels=['A', 'B', 'C', 'D', 'E'], colors=colors_pie_dash,
        autopct='%1.1f%%', startangle=90)
ax4.set_title('Composition', fontweight='bold')

# 5. Box Plot
ax5 = fig.add_subplot(gs[2, 0])
data_box_dash = [np.random.randn(100) + i for i in range(4)]
ax5.boxplot(data_box_dash, labels=['A', 'B', 'C', 'D'], patch_artist=True)
ax5.set_title('Distribution by Category', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Bar Chart
ax6 = fig.add_subplot(gs[2, 1])
categories_bar_dash = ['A', 'B', 'C', 'D', 'E']
values_bar_dash = [23, 45, 56, 78, 32]
ax6.bar(categories_bar_dash, values_bar_dash, color='steelblue', alpha=0.7)
ax6.set_title('Ranking', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Heatmap
ax7 = fig.add_subplot(gs[2, 2])
data_heat_dash = np.random.rand(5, 5)
sns.heatmap(data_heat_dash, annot=True, cmap='coolwarm', ax=ax7, cbar=True)
ax7.set_title('Correlation Matrix', fontweight='bold')

plt.suptitle('Comprehensive Data Visualization Dashboard',
             fontsize=16, fontweight='bold', y=0.995)
plt.show()

print("\n" + "=" * 80)
print("END OF CODE COLLECTION")
print("=" * 80)