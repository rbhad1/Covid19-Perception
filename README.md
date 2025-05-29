# CS 7641 Group 16 Covid-19 Project

## GitHub Repository Guide
/codes/                        # All modeling code, evaluation scripts, and visualizations
├── data_cleaning.ipynb       # Initial data cleaning and preprocessing (first half of columns)
├── second_half_ohe.ipynb     # Initial data cleaning and preprocessing (second half of columns)
├── raw+smote_eda.ipynb       # EDA on raw and SMOTE data after MCA, including tSNE and UMAP
├── mca.ipynb                 # Applies Multiple Correspondence Analysis (MCA)
├── log_regression.ipynb     # Logistic Regression model training and evaluation
├── random_forest.ipynb      # Random Forest model training and evaluation
├── random_forest.py         # Pipeline for Random Forest implementation and evaluation
├── smote.ipynb              # Generates SMOTE-augmented dataset
├── SVM_final.ipynb          # SVM model training and evaluation
├── h_clustering.ipynb       # Hierarchical clustering and dendrogram generation
└── clusters/                # Predicted clusters for k=3 used in Fowlkes-Mallow evaluation

/data/                        # All relevant data (CSV) files
├── Rawdata.csv                       # Unprocessed raw survey dataset
├── final_df.csv                      # Fully processed, one-hot encoded (no target)
├── final_df_with_target.csv         # Fully processed, one-hot encoded (with target)
├── first_half_ohe.csv               # One-hot encoded data (first half of features)
├── second_half_ohe.csv              # One-hot encoded data (second half of features)
├── df_smote_with_target.csv         # SMOTE-augmented, fully processed (with target)
├── mca_components.csv               # MCA-transformed dataset, 75% variance
├── mca_components_smote.csv        # MCA-transformed SMOTE dataset
├── mca_components_smote_75.csv     # MCA SMOTE dataset, 75% variance
└── mca_components_smote_95.csv     # MCA SMOTE dataset, 95% variance

/docs/                        # For GitHub Pages integration (Jekyll template, assets, etc.)
