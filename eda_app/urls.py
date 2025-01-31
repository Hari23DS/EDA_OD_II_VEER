from django.urls import path
from . import views
 
urlpatterns = [
    path('', views.upload_file, name='upload_file'),  # Home page for file upload
    path('eda/', views.eda_analysis, name='eda_analysis'),  # Page for EDA result
    path('imputation_choice/', views.handle_imputation_choice, name='handle_imputation_choice'),  # Yes/No Imputation Page
    path('choose_imputation/', views.perform_custom_imputation, name='choose_imputation'),  # Basic/MICE Imputation Page
    path('statistical_test/', views.statistical_test, name='statistical_test'),  # Page for Statistical Test Results
]

