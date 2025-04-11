# -- Tools for Data Analysis with Plotly Visualizations --

# Libraries
import pandas as pd
import numpy as np
import logging
import sys
import os

# File handling
from pathlib import Path
from warnings import filterwarnings
from scipy.stats import zscore
from scipy.stats import gaussian_kde

# Suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

# Color palette for visualizations
PLOTLY_COLORS = px.colors.qualitative.Plotly
CUSTOM_COLORS = [
    '#023050', '#0080b6', '#0095c7', '#90a4ae', '#6a3d9a', '#8f4f4f', '#e31a1c',
    '#e85d10', '#ff8210', '#ff9c35'
]

# --- Functions ---
# 1. Function to load data from .csv or .xlsx files
def load_data(file_path: str | Path) -> dict:
    """
    Loads a dataset from .csv or .xlsx file and returns structured dictionary.
    Maintains original functionality with improved error handling.
    """
    file_path = Path(file_path).expanduser().resolve()
    
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == ".xlsx":
            with pd.ExcelFile(file_path, engine="openpyxl") as xls:
                sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
            logging.info(f"Excel file loaded. Sheets: {list(sheets.keys())}")
            return {"data": sheets, "metadata": {"file": str(file_path), "sheets": list(sheets.keys())}}
        
        elif file_extension == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            logging.info(f"CSV file loaded. Shape: {df.shape}")
            return df
        
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise RuntimeError(f"Error loading file: {e}")


# 2. 
def display_info(df, plot_numeric: bool = True):
    """
    Enhanced DataFrame information display with optional Plotly visualizations
    """
    print("\n◇ Dataset Info:")
    print(df.info())
    
    print("\n◇ Descriptive Statistics:")
    print(df.describe())
    
    if plot_numeric and isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            print("\n◇ Numeric Distributions:")
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
                fig.show()


# 3. Function to handle missing values with interactive visualization
def handle_missing_values(df, strategy=None):
    """
    Handles missing values with interactive visualization
    """
    missing = df.isnull().sum()
    print("\n◇ Missing Values Before Treatment:")
    print(missing)
    
    if strategy is None:
        # Visualize missingness
        if missing.sum() > 0:
            fig = px.bar(
                x=missing.index, y=missing.values, 
                labels={'x':'Column', 'y':'Missing Count'},
                title="Missing Values by Column"
            )

            fig.show()
        return df
    
    strategies = {
        "mean": df.fillna(df.mean()),
        "median": df.fillna(df.median()),
        "mode": df.fillna(df.mode().iloc[0]),
        "drop": df.dropna()
    }
    
    if strategy in strategies:
        df = strategies[strategy]
        print("\n🔹 Missing Values After Treatment:")
        print(df.isnull().sum())
        
        # Show remaining missing values
        remaining_missing = df.isnull().sum()
        if remaining_missing.sum() > 0:
            fig = px.bar(x=remaining_missing.index, y=remaining_missing.values,
                        labels={'x':'Column', 'y':'Remaining Missing'},
                        title="Remaining Missing Values After Treatment")
            fig.show()
        return df
    else:
        raise ValueError("Invalid strategy. Use: 'mean', 'median', 'mode', or 'drop'")


# 4. Function to visualize distributions of numeric columns with Plotly
def visualize_plots(dataset, columns=None, plot_types=None, 
                    color='', hue=None, height_per_plot=300, 
                    width=1200, kde=False, bins=30,
                    title=None, title_font_size=16, title_y=0.98, title_x=0.5,
                    density_comparison=False, boxplot_x=None, boxplot_colors=None):
    """
    Enhanced visualization function with ADDED hue support while maintaining all original features
    
    Parameters:
        dataset: Can be DataFrame or dict of DataFrames (original behavior preserved)
        columns: List of columns to plot
        plot_types: Plot types per column
        color: Base color
        hue: NEW - Column name for grouping
        height_per_plot: Subplot height
        width: Total width
        kde: Add density curve
        bins: Number of bins
        title: Main title
        title_font_size: Title font size
        title_y: Title vertical position
        title_x: Title horizontal position
        density_comparison: NEW - If True, creates side-by-side density comparison when hue is used
    """
    
    # Handle dict/DataFrame input
    if isinstance(dataset, pd.DataFrame):
        df_dict = {"Dataset": dataset}  
    else:
        df_dict = dataset.copy()

    plot_data = []
    for table_name, df in df_dict.items():
        for col in (columns if columns is not None else df.select_dtypes(include=np.number).columns):
            if col in df.columns:
                plot_data.append((df, col, table_name))  # Now storing (df, col, table_name)

    # Handle hue categories if specified
    if hue is not None:
        hue_categories = pd.concat([df[hue] for df in df_dict.values() if hue in df]).unique()
        color_sequence = px.colors.qualitative.Plotly
    else:
        hue_categories = [None]
        color_sequence = [color] if color else ['#636EFA']

    # Plot type handling preserved
    if plot_types is None:
        plot_types = ['histogram'] * len(plot_data)
    elif isinstance(plot_types, str):
        plot_types = [plot_types] * len(plot_data)

    # Enhanced grid calculation for comparison mode
    n_cols = 2 if (hue and density_comparison) else 3
    n_rows = (len(plot_data) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[col for _, col, _ in plot_data],
        horizontal_spacing=0.05, vertical_spacing=0.1
    )

    for i, ((df, col, table_name), plot_type) in enumerate(zip(plot_data, plot_types)):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1

        # Handle hue groups
        for j, category in enumerate(hue_categories):
            data = df[col] if category is None else df[df[hue] == category][col]
            display_name = f"{table_name} ({category})" if (hue and len(df_dict) > 1) else str(category) if hue else col

            if plot_type == 'histogram':
                fig.add_trace(
                    go.Histogram(
                        x=data, name=display_name,
                        marker_color=color_sequence[j % len(color_sequence)],
                        nbinsx=bins, opacity=0.7,
                        histnorm='probability density' if kde else None,
                        showlegend=(hue is not None) and (i == 0 or density_comparison),
                        legendgroup=hue
                    ),
                    row=row, col=col_pos
                )

                if kde and hue is None:  # KDE behavior preserved
                    density = gaussian_kde(data.dropna())
                    x = np.linspace(data.min(), data.max(), 200)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=density(x), mode='lines',
                            line=dict(color='#023050', width=2),
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )

            if boxplot_x and boxplot_x in df.columns:
                categories = sorted(df[boxplot_x].dropna().unique())
                
                for k, cat in enumerate(categories):
                    filtered_data = df[df[boxplot_x] == cat][col]
                    
                    fig.add_trace(
                        go.Box(
                            y=filtered_data,
                            x=[str(cat)] * len(filtered_data),
                            name=str(cat),
                            marker_color=(
                                boxplot_colors.get(cat, color_sequence[k % len(color_sequence)])
                                if boxplot_colors else color_sequence[k % len(color_sequence)]
                            ),
                            boxpoints='Outliers',
                            showlegend=(i == 0),
                            legendgroup = boxplot_x
                        ),
                        row=row, col=col_pos
                    )

            elif plot_type == 'violin':
                fig.add_trace(
                    go.Violin(
                        y=data,
                        name=display_name,
                        marker_color=color_sequence[j % len(color_sequence)],
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=(hue is not None) and (i == 0),
                        legendgroup=hue
                    ),
                    row=row, col=col_pos
                )

            # Density comparison mode
            if hue and density_comparison and plot_type == 'histogram':
                density = gaussian_kde(data.dropna())
                x = np.linspace(data.min(), data.max(), 200)
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=density(x),
                        mode='lines',
                        line=dict(color=color_sequence[j % len(color_sequence)], width=2),
                        name=display_name,
                        showlegend=(i == 0),
                        legendgroup=hue
                    ),
                    row=row, col=1  # Always in first column for comparison
                )

    # Layout configuration preserved with enhancements
    layout_config = {
        'height': height_per_plot * n_rows,
        'width': width,
        'showlegend': hue is not None,
        'margin': dict(l=50, r=50, b=50, t=60 if title else 30, pad=4),
        'plot_bgcolor': 'white',
        'legend': {'title': {'text': hue}} if hue else None
    }

    if title:
        layout_config['title'] = {
            'text': f'<b>{title}</b>',
            'y': title_y,
            'x': title_x,
            'font': {'size': title_font_size}
        }
    

    fig.update_layout(**layout_config)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    fig.show()


# 5. Function to detect outliers with enhanced visualization options
def detect_outliers(dataset, tables=None, columns=None, strategy=None, 
                   layout_style='grouped', scale='linear', figsize=(20, 10), color=None):
    """
    Enhanced outlier detection with visualization options
    
    Parameters:
        dataset: Input DataFrame or dictionary of DataFrames
        tables: List of table names (if dataset is a dictionary)
        columns: List of columns to analyze
        strategy: Outlier detection strategy (currently supports IQR)
        layout_style: 'compact' (horizontal) or 'grouped' (vertical, default)
        scale: 'linear', 'log', or 'standard' (z-score normalization)
    
    Returns:
        tuple: (outlier_indexes, outlier_counts, total_outliers)
    """    
    
    # Initialize results containers
    outlier_indexes = {}
    outlier_counts = {}
    total_outliers = 0
    selected_data = {}
    
    # Handle multi-table input
    if isinstance(dataset, dict) and 'data' in dataset:
        if tables is None:
            raise ValueError("Specify tables when working with multiple tables")
        
        for table in tables:
            if table in dataset['data']:
                df = dataset['data'][table]
                for col in columns:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        selected_data[f"{table}_{col}"] = df[col]
    else:
        selected_data = {col: dataset[col] for col in columns if col in dataset.columns 
                        and pd.api.types.is_numeric_dtype(dataset[col])}
    
    if not selected_data:
        raise ValueError("No numeric columns found")
    
    # Quantitative outlier detection (IQR method)
    for name, data in selected_data.items():
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indexes[name] = data[outliers_mask].index.tolist()
        outlier_counts[name] = outliers_mask.sum()
        total_outliers += outlier_counts[name]
    
    # Print summary statistics
    print(f"Total outliers detected: {total_outliers}.")
    print("\nOutliers per feature:\n")
    
    for feature, count in outlier_counts.items():
        pct = (count / len(selected_data[feature])) * 100
        print(f"- {feature}: {count} ({pct:.2f}%)")

    # Prepare plot data based on scale
    plot_data = {}
    
    for name, data in selected_data.items():
        if scale == 'log':
            plot_data[name] = np.log1p(data)
        elif scale == 'standard':
            plot_data[name] = zscore(data)
        else:
            plot_data[name] = data

    # Create visualizations
    if layout_style == 'compact':
        # Horizontal compact layout
        fig = go.Figure()
        
        for i, (name, data) in enumerate(plot_data.items()):
            fig.add_trace(go.Box(
                x=data,
                name=f"{name} (Outliers: {outlier_counts[name]})",
                marker_color=CUSTOM_COLORS[i % len(CUSTOM_COLORS)],
                boxpoints='outliers',
                orientation='h',
                line_width=1.5,
                opacity=0.8
            ))
        
        fig.update_layout(
            title={
                'text': "<b>Outliers Detection</b>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            height=max(600, 50 * len(plot_data)),
            margin=dict(l=150, r=50, b=50, t=80, pad=4),
            showlegend=False,
            xaxis_title="" if scale == 'linear' else f"Value ({scale} scale)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    
    else:
        rows = (len(plot_data) + 2) // 3 # Calculate number of rows needed 
        fig = make_subplots(rows=rows, cols=3, subplot_titles=list(plot_data.keys()))
        
        row = 1
        col = 1
        
        for i, (name, data) in enumerate(plot_data.items()):
            fig.add_trace(go.Box(
                y=data,
                name=name,
                marker_color=color if color else CUSTOM_COLORS[i % len(CUSTOM_COLORS)],
                boxpoints='outliers',
                line_width=1.5,
                opacity=0.8,
                whiskerwidth=0.2,
                showlegend=False
            ), row=row, col=col)
            
            col += 1

            if col > 3:
                col = 1
                row += 1
        
        fig.update_layout(
            title_text="Outlier Detection with Box Plots",
            height=300 * rows,
            width=1000,
            margin=dict(l=40, r=40, b=80, t=100),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        fig.update_yaxes(title_text="Valor", showgrid=True, gridcolor='lightgrey')

        fig.show()

    return outlier_indexes, outlier_counts, total_outliers


# 6. Function to plot categorical distributions with Plotly    
def plot_categorical_distribution(df_dict, columns, hue=None, cols=3, figsize=None, 
                                  show_xlabel=False, show_ylabel=True):
    """
    Generates interactive bar plots for multiple categorical variables from multiple DataFrames using Plotly.
    Displays percentages instead of counts and positions text outside the bars without overlapping.

    Parameters:
    - df_dict: Can be either:
        1) A dictionary where keys are table names and values are DataFrames.
        2) A single DataFrame.
    - columns: List of categorical column names to plot.
    - hue: Categorical variable to group by (e.g., 'class').
    - cols: Number of columns in the subplot grid.
    - figsize: Tuple (width, height) defining figure size. If None, it is dynamically adjusted.
    - show_xlabel: Boolean, whether to show x-axis labels.
    - show_ylabel: Boolean, whether to show y-axis labels.
    """

    # If a single DataFrame is provided, wrap it in a dictionary
    if isinstance(df_dict, pd.DataFrame):
        df_dict = {"Dataset": df_dict}  

    plot_data = []

    # Collect valid columns from each dataset
    for table_name, df in df_dict.items():
        for col in columns:
            if col in df.columns:
                plot_data.append((df, col)) 

    num_plots = len(plot_data)
    rows = -(-num_plots // cols)  # Ceiling division without using math.ceil()

    # Dynamic figure sizing with more space for more columns
    if figsize is None:
        width = max(800, cols * 350)  # Increased from 300 to 350 per column
        height = max(400, rows * 350)  # Increased from 300 to 350 per row
    else:
        width, height = figsize[0] * 40, figsize[1] * 40  

    # Create subplots with adjusted spacing
    fig = make_subplots(
        rows=rows, cols=cols, 
        subplot_titles=[title for _, title in plot_data],
        horizontal_spacing=0.15 if cols > 2 else 0.1,  # More space for more columns
        vertical_spacing=0.05 if rows > 1 else 0.1    # More space for more rows
    )   

    # Generate bar charts with percentage labels
    for i, (df, title) in enumerate(plot_data):
        row = (i // cols) + 1
        col = (i % cols) + 1

        if hue and hue in df.columns:
            grouped = df.groupby([title, hue]).size().reset_index(name='count')
            grouped['Percentage'] = grouped.groupby(title)['count'].apply(lambda x: x / x.sum() * 100)
        else:
            grouped = df[title].value_counts(normalize=True).reset_index()
            grouped.columns = [title, 'Percentage']
            grouped['Percentage'] *= 100

        # Format labels
        text_labels = [f"{p:.1f}%" for p in grouped['Percentage']]

        # Define color
        colors = ['#023050', '#0080b6'] if hue else ['#023050']

        # Create bar chart
        for j, hue_value in enumerate(grouped[hue].unique() if hue else [None]):
            subset = grouped[grouped[hue] == hue_value] if hue else grouped
            
            trace = go.Bar(
                y=subset[title], 
                x=subset['Percentage'], 
                orientation='h',
                text=text_labels,
                textposition='outside',
                marker=dict(color=colors[j % len(colors)]),
                name=str(hue_value) if hue else None
            )
            
            fig.add_trace(trace, row=row, col=col)

    # Update layout
    fig.update_layout(
        height=height, 
        width=width, 
        showlegend=True if hue else False,
        margin=dict(l=50, r=50, b=80, t=80, pad=10),
        title_font_size=14,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    # Adjust subplot title font size
    title_font_size = 12 if cols <= 3 else 10 if cols <= 5 else 8
    fig.update_annotations(font_size=title_font_size)

    # Adjust axis ranges
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            fig.update_xaxes(
                range=[0, 110],  # Leave 10% extra space for text
                row=i, col=j
            )

    # Adjust axis labels visibility
    if not show_xlabel:
        fig.update_xaxes(title_text="")

    if not show_ylabel:
        fig.update_yaxes(title_text="")

    fig.show()


# 7. Function to save data as .xlsx or .csv without modifying the original dataset
def save_data(data, output_path, explicit_path=False):
    """
    Saves the dataset as an Excel (.xlsx) or CSV (.csv) file without modifying the original dataset.

    This function takes a dictionary containing multiple sheets (DataFrames) and saves them into a new file. 
    If the dataset contains multiple sheets, it will be saved as an Excel file with separate sheets. 
    If a CSV format is specified, only the first sheet will be saved.

    Parameters:
        data (dict): Dictionary containing the dataset.
            - Must include a "data" key where sheet names are keys and DataFrames are values.
        output_path (str): File path where the dataset will be saved (.xlsx or .csv).

    Returns:
        None: Writes the output file to the specified path.

    Raises:
        ValueError: If the dataset structure is incorrect or contains no valid sheets.
        RuntimeError: If an error occurs while saving the file.

    Logs:
        - Logs a success message when the file is saved.
        - Logs a warning if a sheet is not a DataFrame and is ignored.
        - Logs an error if saving fails.

    Example:
        save_data(dataset, "processed_data.xlsx")

    Raises:
    - ValueError: If the dataset is not structured correctly or contains no valid sheets.
    - RuntimeError: If an error occurs while saving the file.
    """
    
    # Check if the dataset contains multiple sheets
    if not isinstance(data, dict) or 'data' not in data:
        raise ValueError(
            "The dataset must be a dictionary containing 'data' with the sheets."
        )

    sheets = data['data']  # Extracting only the sheets
    if not sheets:
        raise ValueError("The dataset does not contain any sheets to save.")

    file_extension = os.path.splitext(output_path)[1].lower()

    try:
        if file_extension == ".xlsx":
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in sheets.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        logging.warning(
                            f"❌ Warning: '{sheet_name}' is not a DataFrame and was ignored."
                        )
            
            if explicit_path:
                logging.info(f"Excel file successfully saved at: {output_path}")
            else: 
                logging.info(f"Excel file successfully saved.")

        elif file_extension == ".csv":
            first_sheet_name = list(sheets.keys())[0]  # Taking the first sheet
            sheets[first_sheet_name].to_csv(output_path, index=False)
            logging.info(f"CSV file successfully saved at: {output_path}")

        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")

    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise RuntimeError(f"Error saving file: {e}")

