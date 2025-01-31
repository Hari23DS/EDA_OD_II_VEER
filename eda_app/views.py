from django.shortcuts import render
import pandas as pd
from .forms import UploadFileForm
from .utils.imputation import auto_impute, impute_missing_values
from .utils.statistical_tests import feature_selection_stats
 
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
        df = pd.read_json(request.session.get('data'))
       
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
 
        df = pd.read_json(request.session.get('data'))
 
        if user_choice == 'yes':
            # Apply RandomForest-based imputation
            df_imputed = auto_impute(df)
            request.session['data'] = df_imputed.to_json()
 
            # Perform statistical testing
            stats_results = feature_selection_stats(df_imputed, request.session.get('target_column'))
            return render(request, 'eda_app/statistical_test.html', {'results': stats_results})
 
        elif user_choice == 'no':
            # User wants to choose a different imputation method
            return render(request, 'eda_app/choose_imputation.html')
 
    return render(request, 'eda_app/imputation_choice.html')
 
# View to perform user-selected imputation (Basic or MICE)
def perform_custom_imputation(request):
    if request.method == 'POST':
        method = int(request.POST.get('method'))  # 1 for Basic, 2 for MICE
        df = pd.read_json(request.session.get('data'))
        df = impute_missing_values(df, method)
        request.session['data'] = df.to_json()
       
        # Perform Statistical Test
        stats_results = feature_selection_stats(df, request.session.get('target_column'))
        return render(request, 'eda_app/statistical_test.html', {'results': stats_results})
 
    return render(request, 'eda_app/choose_imputation.html')
 
# View to display the statistical test results
def statistical_test(request):
    df = pd.read_json(request.session.get('data'))
    target_column = request.session.get('target_column')
 
    # Perform statistical test
    stats_results = feature_selection_stats(df, target_column)
 
    return render(request, 'eda_app/statistical_test.html', {'results': stats_results})
