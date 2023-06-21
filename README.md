<html>
  <head>
  </head>
  <body style="text-align:justify;">
    <h2 style="text-align:center;"> Concrete Strength Prediction Project </h2>
    <p>
      This project was developed for a fictional company that is involved in concrete production. They sought a solution to predict the strength of the concrete produced in order to improve the quality of the product and increase the efficiency of the production process.
    </p>
    <p>
      Such prediction is crucial for the company as it allows for the identification of potential issues before mass production, thereby avoiding financial losses from poor productions and ensuring customer satisfaction. In addition, the correct prediction of concrete strength allows for the optimization of inputs usage, reducing costs and increasing the company's profitability. The database used is available on Kaggle at the following link: <a href="https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength">https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength</a>
    </p>
    <p>
      In this project, various regression algorithms were evaluated and the best result was obtained using the XGBoost Regressor, with an incredible accuracy of 86.91%. This repository includes the complete source code of the solution, divided into two notebooks:
    </p>
    <ul>
      <li>
        The first notebook performs an exploratory data analysis and data preparation for use, using the pandas, numpy, matplotlib, and seaborn libraries.
      </li>
      <li>
        The second notebook presents the construction and evaluation of the machine learning models, using the sci-kit learn and xgboost libraries, as well as the train_test_split algorithms (LinearRegression, Ridge, Lasso, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, SVR), sci-kit learn metrics (mean_absolute_error, median_absolute_error, r2_score), scipy's stats and zscore, sci-kit learn's KFold from the model selection, and sci-kit learn's cross_val_score. The best performing algorithm was XGBRegressor from xgboost.
      </li>
    </ul>
    <p>
      In addition to the notebooks, there are also two available data files:
    </p>
    <ul>
      <li>
        The original file downloaded from Kaggle in CSV format.
      </li>
      <li>
        A file in parquet format with the cleaned and ready-to-use data for the machine learning model.
      </li>
    </ul>
    <p>
      If you are interested in machine learning and data science, feel free to clone this repository and try it out for yourself. I am also always open to feedback and contributions to further enhance this project.
    </p>
  </body>
</html>
