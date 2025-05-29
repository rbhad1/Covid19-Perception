# Prediction of COVID-19 Perception Using Personal Financial Circumstances

To explore the relationship between perception of COVID and personal finance circumstance, we use machine learning methods to analyze data from [Harvard’s A Survey on Health Care Access During the COVID-19 Pandemic (June 2020)](https/dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XCKM0V). 

For data preprocessing, we perform one-hot encoding to convert categorical variables into a numerical form, multiple correspondance analysis (MCA) to reduce the number of features, and SMOTE since the initial data had severe class imbalance. We first performed Hierarchical Clustering to determine how our data naturally clusters and if it clusters into sentiment categories of "Not Worried", "Somewhat Worried", and "Very Worried". We also ran Logistic Regression to determine if the data our data is linear or nonlinearlly organized. Both models showed our dataset has complex, nonlinear relationships, indicating that there is not a straightforward relationship between COVID-19 perception and personal finance situation. This prompted us to run models that can handle nonlinearly in data, namely Random Forest and Support Vector Machine (SVM). We concluded that the relationship between financial circumstances and attitudes toward COVID-19 may be more complex. While personal financial situations do play a role, attitudes are shaped by a variety of intersecting factors, highlighting the need for a more nuanced understanding of public health behavior.

Below is a summary of the results:


<table cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th rowspan="2" style="border-bottom: 3px solid black; border-right: 3px solid black;">Metric</th>
      <th colspan="2" style="border-bottom: 3px solid black; border-right: 3px solid black;">Logistic Regression</th>
      <th colspan="2" style="border-bottom: 3px solid black; border-right: 3px solid black;">Support Vector Machine</th>
      <th colspan="2" style="border-bottom: 3px solid black; border-right: 3px solid black;">Random Forest</th>


    </tr>
    <tr>
      <th style="border-bottom: 1px solid black; border-right: 1px solid black;">Worst Performing: SMOTE + 75% MCA</th>
      <th style="border-bottom: 3px solid black; border-right: 3px solid black;">Best Performing: SMOTE</th>
      <th style="border-bottom: 1px solid black; border-right: 1px solid black;">Worst Performing: Raw Data</th>
      <th style="border-bottom: 3px solid black; border-right: 3px solid black;">Best Performing: SMOTE</th>
      <th style="border-bottom: 1px solid black; border-right: 1px solid black;">Worst Performing: Raw Data</th>
      <th style="border-bottom: 3px solid black; border-right: 3px solid black;">Best Performing: SMOTE</th>


    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 3px solid black; border-top: 3px solid black;">Accuracy</td>
      <td style="border-top: 3px solid black; border-right: 1px solid black">0.51</td>
      <td style="border-top: 3px solid black; border-right: 3px solid black">0.63</td>
      <td style="border-right: 1px solid black; border-top: 3px solid black;">0.57</td>

      <td style="border-top: 3px solid black; border-right: 3px solid black">0.68</td>
      <td style="border-right: 1px solid black; border-top: 3px solid black">0.58</td>

      <td style="border-top: 3px solid black; border-right: 3px solid black">0.73</td>

    </tr>
  </tbody>
</table>

For full project details on methodology, results, analysis, and future steps, please view [Project Report]()


[1] E. Díaz McConnell, C. M. Sheehan, and A. Lopez, “An intersectional and social determinants of health framework for understanding Latinx psychological distress in 2020: Disentangling the effects of immigration policy and practices, the Trump Administration, and COVID-19-specific factors,” Journal of Latinx Psychology, vol. 11, no. 1, pp. 1–20, 2023.
[2] D. Warmath, G. E. O’Connor, C. Newmeyer, and N. Wong, “Have I saved enough to social distance? The role of household financial preparedness in public health respgit puonse,” J. Consum. Aff., vol. 56, no. 1, pp. 319–338, 2022.
[3] J. S. Trueblood, A. B. Sussman, D. O’Leary, and W. R. Holmes, “Financial constraint and perceptions of COVID-19,” Sci. Rep., vol. 13, no. 1, p. 3432, 2023.


## GitHub Repository Structure
```
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
```
## How to Run the Models

To access and run the models, clone the Git repo, open in Virtual Studio Code (or equivalent), and run the desire file.

