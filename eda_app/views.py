from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .forms import UploadFileForm
from .utils.imputation import auto_impute, impute_missing_values
from .utils.statistical_tests import feature_selection_stats
import io
from IPython.display import display
import base64

# Function to generate dataset overview
def dataset_overview(df):
    """
    Generate an overview of the dataset in a structured, column-wise format.
    """
    num_variables = df.shape[1]
    num_observations = df.shape[0]
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / (num_variables * num_observations)) * 100 if num_observations > 0 else 0
    duplicate_rows = df.duplicated().sum()
    total_size = df.memory_usage(deep=True).sum() / 1024  # Convert to KB
    avg_record_size = (total_size * 1024 / num_observations) if num_observations > 0 else 0  # Bytes
 
    # Variable types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    text_cols = df.select_dtypes(include=['string']).columns.tolist()
 
    # Collect column data types
    column_data_types = df.dtypes.to_frame().reset_index()
    column_data_types.columns = ["Feature", "Data Type"]
 
    # Organize data in row-wise format
    overview = [
        ["Shape of the dataset", f"{df.shape[0]} rows, {df.shape[1]} columns", "", ""],
        ["Number of variables", num_variables, "", ""],
        ["Number of observations", num_observations, "", ""],
        ["Missing cells", missing_cells, "", ""],
        ["Missing cells (%)", round(missing_percentage, 1), "", ""],
        ["Duplicate rows", duplicate_rows, "", ""],
        ["Duplicate rows (%)", round((duplicate_rows / num_observations) * 100, 1) if num_observations > 0 else 0, "", ""],
        ["Total size in memory (KB)", round(total_size, 2), "", ""],
        ["Average record size in memory (Bytes)", round(avg_record_size, 1), "", ""],
        ["Numeric", "", len(numeric_cols), ""],
        ["Categorical", "", len(categorical_cols), ""],  
        ["Text", "", len(text_cols), ""],  
    ]
 
    # Add column data types correctly
    for _, row in column_data_types.iterrows():
        overview.append([row["Feature"], "", "", row["Data Type"]])
 
    return overview
 
# File Upload View
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            try:
                df = pd.read_csv(uploaded_file)  # Read CSV File
                request.session['data'] = df.to_json()  # Store Data in Session
                request.session['filename'] = uploaded_file.name  # Store filename for reference
               
                # Extract column names for selection
                columns = list(df.columns)
               
                return render(request, 'eda_app/select_target.html', {'columns': columns})
            except Exception as e:
                return render(request, 'eda_app/upload.html', {'form': form, 'error': str(e)})
    else:
        form = UploadFileForm()
    return render(request, 'eda_app/upload.html', {'form': form})
 
# View for performing EDA
def eda_analysis(request):
    if request.method == 'POST':
        selected_column = request.POST.get('target_column')
        df = pd.read_json(io.StringIO(request.session.get('data')))
       
        # Step 1: Generate dataset overview
        overview = dataset_overview(df)
 
        # Store target column in session
        request.session['target_column'] = selected_column
 
        # Step 2: Ask user if they are satisfied with the dataset overview before imputation
        return render(request, 'eda_app/imputation_choice.html', {
            'overview': overview,
            'target_column': selected_column
        })
 
    return render(request, 'eda_app/select_target.html', {})
 
# View to handle user choice after dataset overview
def handle_imputation_choice(request):
    if request.method == 'POST':
        user_choice = request.POST.get('imputation_choice')
 
        df = pd.read_json(io.StringIO(request.session.get('data')))
 
        if user_choice == 'yes':
            # Apply RandomForest-based imputation
            df_imputed = auto_impute(df)
            request.session['data'] = df_imputed.to_json()
            target_column = request.session.get('target_column')
 
            # Perform statistical testing
            stats_results = feature_selection_stats(df_imputed, target_column)
            plot_images = eda_analysis_final(df, target_column, stats_results)

            return render(request, 'eda_app/statistical_test.html', {'results': stats_results, 'output_data' : plot_images})
 
        elif user_choice == 'no':
            # User wants to choose a different imputation method
            return render(request, 'eda_app/choose_imputation.html')
 
    return render(request, 'eda_app/imputation_choice.html')
 
# View to perform user-selected imputation (Basic or MICE)
def perform_custom_imputation(request):
    if request.method == 'POST':
        method = int(request.POST.get('method'))  # 1 for Basic, 2 for MICE
        df = pd.read_json(io.StringIO(request.session.get('data')))
        df = impute_missing_values(df, method)
        request.session['data'] = df.to_json()
        target_column = request.session.get("target_column")
       
        # Perform Statistical Test
        stats_results = feature_selection_stats(df, request.session.get('target_column'))
        plot_images = eda_analysis_final(df, target_column, stats_results)

        return render(request, 'eda_app/statistical_test.html', {'results': stats_results, 'output_data' : plot_images})
 
    return render(request, 'eda_app/choose_imputation.html')
 
# View to display the statistical test results
def statistical_test(request):
    df = pd.read_json(request.session.get('data'))
    target_column = request.session.get('target_column')
 
    # Perform statistical test
    stats_results = feature_selection_stats(df, target_column)
    plot_images = eda_analysis_final(df, target_column, stats_results)

 
    return render(request, 'eda_app/statistical_test.html', {'results': stats_results, 'plot_images' : plot_images})





def save_plot_to_base64(fig):
    """Save the plot to a base64 image and return the encoded string."""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    print("inside the save_plot_to_base64")
    return f"data:image/png;base64,{img_base64}"


# def eda_analysis_final(df, target_column, stats_results=None):
#     """Performs EDA and returns plots as base64 images for Django templates."""
#     graph_suggestions = stats_results["suggested_features"]
#     # stats_results = feature_selection_stats(df, target_column)["statistical_results"]
#     num_cols = df.select_dtypes(include=np.number).columns.tolist()
#     cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
#     if target_column in cat_cols:
#         cat_cols.remove(target_column)
    
#     if graph_suggestions:
#         num_cols = [col for col in num_cols if col in graph_suggestions]
#         cat_cols = [col for col in graph_suggestions if col in cat_cols]

#     plot_images = []  # Store generated plots as base64 images

#     for col in cat_cols + num_cols:

#         print(f"\n--- Analyzing Column: {col} ---")
#         stat_data = stats_results.get(col, None)
#         if isinstance(stat_data, dict):
#             stat_df = pd.DataFrame([stat_data])
#             display(stat_df.style.set_caption(f"Statistical Test Results for {col}").set_table_styles(
#                 [{'selector': 'caption', 'props': [('color', 'black'), ('font-size', '16px')]}]))
#         else:
#             print("No test performed")
#         fig, axes = plt.subplots(1, 3 if col in num_cols else 1, figsize=(15, 5))

#         if col in cat_cols:
#             sns.countplot(data=df, x=col, ax=axes)
#             axes.set_title(f"Bar Chart: {col}")
#         else:
#             sns.histplot(df[col], kde=True, ax=axes[0])
#             axes[0].set_title(f"Histogram: {col}")
            
#             sns.boxplot(y=df[col], ax=axes[1])
#             axes[1].set_title(f"Box Plot: {col}")

#             sns.kdeplot(df[col], fill=True, ax=axes[2])
#             axes[2].set_title(f"KDE Plot: {col}")

#         if target_column and target_column in df.columns:
#             if col in num_cols:
#                 sns.regplot(data=df, x=col, y=target_column, ax=axes[3])
#                 axes[3].set_title(f"Regression Plot: {col} vs {target_column}")
#             elif col in cat_cols:
#                 sns.countplot(data=df, x=col, hue=target_column, ax=axes[3])
#                 axes[3].set_title(f"Count Plot: {col} by {target_column}")

#         # Save plot as base64 image
#         plot_images.append(save_plot_to_base64(fig))
#         plt.close(fig)  # Close the figure to free memory

#     return plot_images  # Return list of image URLs






def eda_analysis_final(df, target_column, stats_results):
    """Performs EDA and returns statistical results and plots as base64 images."""
    graph_suggestions = stats_results.get("suggested_features", [])
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    stats_results = stats_results["statistical_results"]
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    if graph_suggestions:
        num_cols = [col for col in num_cols if col in graph_suggestions]
        cat_cols = [col for col in graph_suggestions if col in cat_cols]

    output_data = []  # Store results and plots for each column

    for col in cat_cols + num_cols:
        print(f"\n--- Analyzing Column: {col} ---")  # Debugging

        # Get statistical test results for the column
        stat_data = stats_results.get(col, None)
        if isinstance(stat_data, dict):
            stat_df = pd.DataFrame([stat_data])
            stat_table_html = stat_df.style.set_caption(f"Statistical Test Results for {col}").set_table_styles(
                [{'selector': 'caption', 'props': [('color', 'black'), ('font-size', '16px')]}]
            ).to_html()

        else:
            stat_table_html = "<p>No test performed for this column.</p>"

        # Generate plots for the column
        plot_images = []
        num_subplots = 4 if target_column and col in num_cols else 3 if col in num_cols else 1
        fig, axes = plt.subplots(1, num_subplots, figsize=(20, 5))

        if num_subplots == 1:  # Handle single-axis case
            axes = [axes]

        if col in cat_cols:
            sns.countplot(data=df, x=col, ax=axes[0])
            axes[0].set_title(f"Bar Chart: {col}")
        else:
            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f"Histogram: {col}")

            sns.boxplot(y=df[col], ax=axes[1])
            axes[1].set_title(f"Box Plot: {col}")

            sns.kdeplot(df[col], fill=True, ax=axes[2])
            axes[2].set_title(f"KDE Plot: {col}")

            if target_column and target_column in df.columns:
                sns.regplot(data=df, x=col, y=target_column, ax=axes[3])
                axes[3].set_title(f"Regression Plot: {col} vs {target_column}")

        # Save the figure as Base64 image
        plot_images.append(save_plot_to_base64(fig))
        plt.close(fig)

        # Append the results and plots for the current column
        output_data.append({
            "column": col,
            "stat_table": stat_table_html,
            "plots": plot_images
        })

    return output_data









# def eda_analysis(df, target_column=None, graph_suggestions=None):
#     stats_results = feature_selection_stats(df, target_column)["statistical_results"]
#     num_cols = df.select_dtypes(include=np.number).columns.tolist()
#     cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
#     if target_column in cat_cols:
#         cat_cols.remove(target_column)
#     if graph_suggestions:
#         num_cols = [col for col in num_cols if col in graph_suggestions]
#         cat_cols = [col for col in cat_cols if col in graph_suggestions]
#     print("\n### Column-wise EDA Analysis ###")
#     all_cols = cat_cols + num_cols
#     for col in all_cols:
#         print(f"\n--- Analyzing Column: {col} ---")
#         stat_data = stats_results.get(col, None)
#         if isinstance(stat_data, dict):
#             stat_df = pd.DataFrame([stat_data])
#             display(stat_df.style.set_caption(f"Statistical Test Results for {col}").set_table_styles(
#                 [{'selector': 'caption', 'props': [('color', 'black'), ('font-size', '16px')]}]))
#         else:
#             print("No test performed")
#         fig, axes = plt.subplots(1, 4, figsize=(24, 5))
#         if col in cat_cols:
#             print("Frequency Table:")
#             print(df[col].value_counts())
#             sns.countplot(data=df, x=col, ax=axes[0])
#             axes[0].set_title(f"Bar Chart: {col}")
#             axes[0].tick_params(axis='x', rotation=90)
#         elif col in num_cols:
#             sns.histplot(df[col], kde=True, ax=axes[0])
#             axes[0].set_title(f"Histogram of {col}")
#             sns.boxplot(y=df[col], ax=axes[1])
#             axes[1].set_title(f"Box Plot of {col}")
#             sns.kdeplot(df[col], fill=True, ax=axes[2])
#             axes[2].set_title(f"KDE Plot of {col}")
#         if target_column and target_column in df.columns:
#             if col in num_cols:
#                 sns.regplot(data=df, x=col, y=target_column, ax=axes[3])
#                 axes[3].set_title(f"Regression Plot: {col} vs {target_column}")
#             elif col in cat_cols:
#                 sns.countplot(data=df, x=col, hue=target_column, ax=axes[3])
#                 axes[3].set_title(f"Count Plot: {col} by {target_column}")
#         plt.tight_layout()
#         plt.show()
#     ## Correlation Heatmap for Numeric Columns
#     if len(num_cols) > 1:
#         plt.figure(figsize=(10, 6))
#         corr_matrix = df[num_cols].corr()
#         sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
#         plt.title("Correlation Heatmap (Numerical Variables)")
#         plt.show()
#     print("\nEDA Analysis Completed.")
