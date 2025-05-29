# Prediction of COVID-19 Perception Using Personal Financial Circumstances

With its outbreak in March 2020 and millions of cases, the COVID-19 pandemic prompted public health measures like social distancing and mask mandates [2]. Research shows that adherence to these policies is shaped by individual perceptions of COVID-19 severity and personal risk [2]. While studies have examined the links between socioeconomic status, disease transmission, and risk behaviors, little research has explored how personal financial circumstances influence perceptions of illnesses like COVID-19 [3]. Preliminary findings suggest financial anxiety heightens psychological distress related to the virus, while financial preparedness may reduce urgency to follow social distancing guidelines [2,3]. Financial constraints may shape beliefs about personal infection risk and virus spread, potentially more than expected factors like political partisanship or local case severity [1,3].

To explore this relationship, we analyze data from [Harvard’s A Survey on Health Care Access During the COVID-19 Pandemic (June 2020)](https/dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XCKM0V). This dataset includes responses on health care access, financial situations, public health policy views, and COVID-19 perceptions.

For data preprocessing, we perform one-hot encoding to convert categorical variables into a numerical form, multiple correspondance analysis (MCA) to reduce the number of features, and SMOTE since the initial data had severe class imbalance. We first performed Hierarchical Clustering to determine how our data naturally clusters and if it clusters into sentiment categories of "Not Worried", "Somewhat Worried", and "Very Worried". We also ran Logistic Regression to determine if the data our data is linear or nonlinearlly organized. Both models showed our dataset has complex, nonlinear relationships, indicating that there is not a straightforward relationship between COVID-19 perception and personal finance situation. This prompted us to run models that can handle nonlinearly in data, namely Random Forest and Support Vector Machine (SVM). We concluded that the relationship between financial circumstances and attitudes toward COVID-19 may be more complex. While personal financial situations do play a role, attitudes are shaped by a variety of intersecting factors, highlighting the need for a more nuanced understanding of public health behavior.

Below is a summary of the methodology and results:



For full project details on methodology, results, analysis, and future steps, please view [Project Report](https://production-gradescope-uploads.s3-us-west-2.amazonaws.com/uploads/text_file/file/814582553/Prediction_of_COVID-19_Perception_Using_Personal_Financial_Circumstances___CS_7641_Group_16.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAV45MPIOWUYFKL2NP%2F20250529%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250529T165903Z&X-Amz-Expires=10800&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJIMEYCIQDNOx9q1F7lWb620T4SsbKHTKI9bJKHpdVldS%2B1UoQ97wIhAL0R%2F7FvLMXoEK2xDWMAVItCX2h3bVb4Irg2c5kc9zQUKsMFCJL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMNDA1Njk5MjQ5MDY5IgyvGFyl%2BRx9wXYc3VgqlwU7Q0KGm%2BJzvGuI5QX7OBoPXLG2D%2BlR90CTH2hp9DDm5JPCIwNg8z7ReT7lElHgjcO0Lnpwq78RzqS3%2FDbSYJFicJ%2BMgAqKMfJqatJd9cfoLPOkU5nonCMAxvMHiCwRpobOWnZdM60OzJofHaRJDC7%2BbTp1mNyZAKG81Sd82gVTKp%2Bcxr9zBRa3ZEYqMgEaaG4UGcm6evU0ajo3u6l6XsONAHHthCMMF1lAo0fkGzMeqmykuCIE%2BVguiL%2Bv7jQN7R18i1HjIGayAyhTmadKwRRHMNpji1CdbKPZqf2tnj1ETCymnhSjF54d2YaLq%2FSQ%2B%2FrQ%2B1dxFGPyBTiQOCXM81Oo7cInfLWet9EotSbAeZFZpazD4pEIxgtL3JA5PSbnOiSalPtnLi8L6AiSiE2NZQfyc77FQ0eGJHGXraT7K1VSbYhMiXTkG6awYQcY%2FXPvG%2FhGi3puMK8tqOYGYU21OB5r%2FIzRkmmQ09SPUCc1o8ULLfmtpLHsQTmuYDYBiWV8AkycztZEyUNham19eI%2BD27y6k4eUhK4I3rMEjxKXHcMVF0ehRxZQ8GtqafTxTh1lrpWOjGxY%2BychOQJ50jKRgLiuC3lULX6XOEm4m8bdnF%2B%2BINaLav7Y28fpsIJZT7eSyTZ%2B5%2BMzO3MErPmfJ5Ko2RYqpL0T7e2xvZpvBePryhrKM9sSYcAq19V61n69OexMpDDrpMa2%2Ff4pU9A3aaxS471E9sZ9TLbHtKkJpOv8GjauTLfpTHQePnYqHaxQTijNUH7cIyKMw6KcYuXZJZKNvghODPbXkiMpc1PYHgZVsGQnQcszXOMoXwmcYmFUIGdkY%2BNoOG5oU7jganRdd62I78ZySSFFHLqePA%2Br6hkW0t0EYoVX9lcDa9Uw8pviwQY6sAHNAK4tohhemKn%2FTDUJMBXSLMYDURTKjBDw6hX67DlLYT4QFaDC%2B5rZ0hc4JnE4R5AZ%2BElsYBk61P2sKa4vGzzaqNB%2B5NAYE5k5cBlnQRCSY2j%2F2WuMNBbiPSnqVZAddkTm%2FvYpXBLsa8uyjleZahBLZMjR%2FG6OYbuadLeT%2Fl0OoGtrI6uhxOiePhRk9c07PBn29URs9Y3fzG2uMAoy5C7ATut61ds1ro0P3YGsBim6CQ%3D%3D&X-Amz-SignedHeaders=host&X-Amz-Signature=edc42dcea1ea03198f2cb085d036392c8616076a055499b5511d1fd6fd8f1e4a)


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

