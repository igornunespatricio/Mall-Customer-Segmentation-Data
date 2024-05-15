# Mall Customer Segmentation Data

This repository is based on [this](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) kaggle dataset. On summary, this dataset contains information (Age, Annual Income, Gender and Spending Score) for 200 customers of a supermarket mall. 

The purporse of this project is to segment those customers using KMeans clustering so this information can be used by the marketing team to plan strategies accordingly. More information can be found in the kaggle dataset link.

A sample of the dataset is provided below.

| CustomerID | Gender | Age | Annual Income (k$)  | Spending Score (1-100) |
|------------|--------|-----|---------------------|------------------------|
|     1      |  Male  |  19 |          15         |           39           |
|     2      |  Male  |  21 |          15         |           81           |
|     3      | Female |  20 |          16         |            6           |
|     4      | Female |  23 |          16         |           77           |
|     5      | Female |  31 |          17         |           40           |


## Data Analysis

This first part of the project contains some quick data analysis using [ydata_profiling](https://github.com/ydataai/ydata-profiling) (former pandas-profiling). This package, as described in the source, provide a one-line Exploratory Data Analysis (EDA) experience in a consistent and fast solution. The full report from this can be found in [this html file](reports\exploratory_data_analysis.html). To run the file, download it to your computer and simply open it.

The code that produced this report is in [the exploratory data analysis notebook](notebooks\exploratory_data_analysis.ipynb). It also contains other charts and tables to help understand the database and give a good baseline to start modeling. In this step, we found that:
* The database doesn't contain null values.
* The Annual Income contains outliers.
* There is no apparent difference when considering segmenting the features by gender.
* Age and Spending Score are correlated (the spending score for customers above 40 drops significantly, we don't see senior customers with high spending score)
* There are apparently 5 clusters as per the image below.

<img src="images\annual_income_spending_score_relation_5_clusters_age_legend.png" alt="scatter plot for annual income and spending score, shows 5 clusters and higher spending score for younger people">


# Modeling
In the modeling part, a pipeline was created with [Scikit-learn](https://scikit-learn.org/stable/) to preprocess and fit KMeans to the dataset. The pipeline take the raw [data](data\Mall_Customers.csv), do some preprocessing steps and fits KMeans in the end. The benefits of using pipelines in machine learning models are that they are more convenient and simplify the process by chaining together multiple steps. They also prevent test data from influencing the training, organize the code, and reduce the risk of errors. More details about the pipeline below.

### Pipeline Steps

1. Input median value for each numerical feature where there are null values.
2. Input most frequent values for each categorical feature (in this case only Gender) where there are null values.
3. Scale numerical features
4. Perform One Hot Encoder with Gender. In this case, since categories are just Male and Female, the one hot encoder will map these values to 0 or 1.
5. Apply PCA with 3 components (pca0, pca1 and pca2).
6. Apply KMeans algorithm.

The pipeline was wrapped up inside a function where we could change the scaler and the KMeans hyperparameters and output measures to compare between different choices. For this purpose, I chose 3 scalers to test: [standard scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler), [minMax scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) and [robust scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) and 2 to 9 number of clusters in the KMeans algorithm. After performing those $3 \times 8 = 24$ pipelines, the chart with the Inertia and the Silhouette scores can be seen below.

<img src="images\inertia_and_silhouette.png" alt="scatter plot for annual income and spending score, shows 5 clusters and higher spending score for younger people">

As seen in the data analysis, 5 clusters would be a good option for this dataset and the inertia and silhouette scores contribute to that choice. But the pipeline selected in this case was the one with robust scaler (to mitigate the outliers effects) and KMeans with 6 clusters.

After fitting the choosen pipeline, we've created the chart below that outputs a 3D scatter plot connecting the data points to their centroids. Each axis correspond to one of the 3 principal components (pca0, pca1 and pca2).

<img src="images\scatter_3D_data_points_clusters.png" alt="3D scatter plot with pca showing data points and clusters connected">

The 3D scatter shows that the clusters are well separated. For better visualization you can download the [html file](reports\3d_scatter_plot_with_cluster_centroids.html), open and play around with it to see their separation.

Also, to understand what are the relationship between the original features (Age, Annual Income, Spending Score and Gender) with the PCa components, a chart was created with the weight of each feature in each PCa component.

<img src="images\pca_component_feature_combination.png" alt="feature weight for each pca component">

We can see, for example, that for pca1 the Age is the most important feature (weight equals 1) while other features are irrelevant in this component. pca0 and pca2 combines Gender and Annual Income.

In the end, we can map personas to those clusters with the help of boxplots for each feature in the original feature space like below.

<img src="images\boxplot_cluster_feature.png" alt="boxplot to help map personas to clusters">

For example, it can be seen from the boxplots that for cluster 1 we have adults between 20~30 years old, with an average annual income and average spending score. On contrast, cluster 2 is for the same age range (20~30), but with a lower annual income and higher spending score. Clusters 3 and 5 are related to wider age ranges with low spending score, but cluster 5 is for lower annual income costumers while cluster 3 is for higher annual income customers.

This information can be useful from a marketing perspective to create more personalized content by focusing in groups of people with similar characteristics found by the Kmeans pipeline model developed in this project.