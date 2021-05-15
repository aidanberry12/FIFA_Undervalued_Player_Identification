# FIFA Undervalued Player Identification and Team Budget Optimization
Author: Aidan Berry </br>
Date: May 14, 2021

## Overview

As a professional soccer club team manager, there is a fixed monetary budget each year to construct the
team. The managers want to build a team to maximize the probability of winning as many games as
possible in the season, while minimizing the cost of acquiring/maintaining the players on the team, or in
general, recruit the highest quality new talent for the lowest possible price. The interesting thing about
professional soccer, is that it is an international sport, so there are many different leagues containing many
teams in each country/continent/region of the world. Throughout all these leagues, there are no shortage
of highly talented players. Some of these players are very well known from their performance throughout
the course of their career, which usually equates to a high monetary value of that player (expensive for
clubs to acquire them). On the other hand, there are also many players that are highly skilled and talented,
but are not as well known, causing them to have a lower realized value on the market. This may be
because they are younger and have not been in the league for as long as more popular players, or that they
have not played on high profile clubs that fans follow closely and have more popularity. These are the
“hidden gem” players that we want to target as a team manager to maximize the utility of the allocated
budget.

Another factor to consider when making player purchasing decisions is the age and future potential of the
player. The career of a soccer player is somewhat short compared to other sports and usually ranges from
about 20-30 years of age before they phase out and lose their edge. It is crucial to build a team that will
not only perform well in the current season, but also in the future seasons to come. Based on this, a club
will want to acquire younger players with high future potential to turn into a star. This is like an
investment for the team; they purchase a young player at a low price before they have developed to their
full potential, and then grow this player on their team over the next few years. These types of purchases
are a risk for any team, as the team cannot be sure if the player will turn out to be a star (good investment)
or if they will be a flop (bad investment). Team managers want to maximize the probability that a young
player will further develop into a key player of the team in the coming years, which is what will be
explored in depth throughout this analysis.

## Datasets Used

The data source that was used for this analysis is the FIFA 21 Complete Player Dataset from [Kaggle](https://www.kaggle.com/stefanoleone992/fifa-21-complete-player-dataset). This dataset is scraped from the video games FIFA 15, FIFA 16, FIFA 17, FIFA 18, FIFA 19, FIFA 20, and FIFA 21 by EA Sports and contains a wide variety of data on all 16,000+ professional soccer players that are in each game. The dataset contains data for all the players in each of the following years: 2015, 2016, 2017, 2018, 2019, 2020, 2021. The features of this dataset include information about each player such as physical attributes (height, weight, etc.), skill attributes (dribbling, shooting, passing, heading, etc.), pricing information (player value, player salary), photos of the players, nationality, and other general player information such as preferred foot, position, international reputation, and contract terms. The 2 core datasets of this analysis are the FIFA 15 and FIFA 21 datasets, where the former is used as a quasi-training set and the latter is used as a quasi-test set. I also used the 2016-2020 datasets for training different model horizons of the response variable (player value), but the 2015 dataset was the only dataset used fully for training and the models’ prediction results are reported on the 2021 data as a current recommendation to team managers.

## Model Architecture

To perform this analysis, a series of supervised regression models were used to identify undervalued and underrated players over various time horizons (from 1 to 6 years in the future) that a team manager should look to pick up in the transfer window to capture the player’s growth curve early in the cycle. Once the optimal regression model is chosen and trained on the 2015 data for each time horizon, the models will then be used to predict the top growth value players looking forward 1-6 years from the current season (2021).

![image](https://user-images.githubusercontent.com/55678487/118347204-ff255500-b506-11eb-8cff-befb119bf6e1.png)

To generate the training set for each model horizon, the datasets for each year were inner joined on the index (sofifa_id) to drop any players not common between the 2 years, and then the difference was taken between the value of the player in the respective years. To get a standardized metric to use as the response variable of the regression problem, the relative percent change in player value was computed for each player that shows up in both years. This relative percent change in the player value (in euros) over the horizon years is the response variable for this analysis, and this was what was being predicted by the regression models being fit later. The “horizon” in this case means the number of years out from the current year to predict the player value change. For example, a horizon 3 model fit on the 2021 data would predict the relative percent change in the player value from 2021 to 2024. If a team is looking to optimize their team performance for 3 years out (in the year 2024) they would recruit the players with the highest predicted percent change in value over this next 3-year horizon. 

## Exploratory Analysis

The boxplot showing the distribution of player values (the response variable in this case) of the 2015 dataset is shown below. From this plot, it becomes obvious that the distribution is strongly right tailed, with many outlier players having values up to 100 million euros. Most of the players in the data have a value of less than 2.5 million euros. Presumably, the “undervalued” players detected by the models will be in this majority group of players at the left tail of the distribution, as the outlier players on the right tail are unlikely to be “undervalued” and have already realized their full potential in their career.

![image](https://user-images.githubusercontent.com/55678487/118347218-1fedaa80-b507-11eb-9ba5-8ddbfe7c2b92.png)

In the figure below, the player overall rating distribution is approximately normally distributed for 2015. This makes sense, as the rating system was likely designed such that there were equal numbers of players on the high and low end of the spectrum. 

![image](https://user-images.githubusercontent.com/55678487/118347231-37c52e80-b507-11eb-9e35-42f78c77accb.png)

To get a more wholistic view of the dataset, principal component analysis (PCA) was performed to produce a reduced representation of the dataset and project the 2015 dataset onto 2 principal axes that capture the most variance. The plot showing the top 2 principal components is shown below, where the purple points represent the non-goalkeeper players, and the yellow points represent the goalkeeper players. Interestingly, the PCA reduced representation shows an obvious clustering of the goalkeepers and the non-goalkeepers with perfect separation. From this, we can conclude that the goalkeeper attributes are significantly different than that of the normal players, thus including the goalkeepers in the model training may introduce unnecessary noise and bias into the models. To mitigate this, the goalkeepers should be removed from the training sets for the regression models.

![image](https://user-images.githubusercontent.com/55678487/118347248-4d3a5880-b507-11eb-93b5-e0b5915b308d.png)

After removing these goalkeepers from the dataset, the updated reduced PCA representation of the top 2 principal components are shown below, with the darkness of the red corresponding to the value of the player (darker red = higher player value). The high value players tend to be located around the outer boundaries of the main cluster of players based on the first 2 principal components, with the highest value players having the largest Euclidean distance from the center of the general player cluster. This is likely caused by the high value players having abnormal game attributes and stats that cause them to be an anomaly based on the “average player”, thus demanding a higher monetary value.

![image](https://user-images.githubusercontent.com/55678487/118347264-5e836500-b507-11eb-9861-85e65b5e08a5.png)

## Feature Engineering and Preprocessing

A generous amount of data preprocessing and data cleaning was required to get the data into a usable form for use in the regression model training. The first step was to drop unnecessary features from the dataset that provide little to no value in the prediction of the player value. Features were also dropped that had missing values in most of the rows. In addition, columns were dropped that had a very large number of possible values (such as team_name), which would cause the dimensionality of the data to increase drastically after one-hot encoding the categorical columns. Next, any categorical columns that are very specific, where certain values may be seen in the test set, but not in the training set were removed. For example, the body_type feature had its possible values change across games over the years, so the one-hot encoded features of the training set would not map to the values of the test set (different years of data), causing the feature to be useless in prediction. It was also decided to drop the wage_eur variable, as it is extremely highly correlated with the player value and ends up being the only significant predictor in the model when it is included (data leakage).

The goalkeepers are significantly different in attribute makeup compared to field players, thus it made sense to remove them from the data for this analysis. The goalkeeper stat columns were dropped as well. Any remaining missing values for the stats were then filled with a 0 value. Missing league rank values were filled with 4 (the worst league rank).

Some further feature engineering that was performed was to parse out the attacking and defensive work rates using string processing. Label encoding was used to map the work rate values (low, medium, and high) to discrete, ordinal values (1, 2, 3), as the higher workrate corresponds to better player performance. Another form of feature engineering performed was mapping the nationality of each player to a continent to reduce the number of possible values using the [pycountry_convert](https://pypi.org/project/pycountry-convert/) Python package. There were some stray country names that were listed in the dataset under a different name than the name list in the package, thus further manual mapping was necessary in these cases. 

Lastly, the data was standardized to zero mean and unit variance to ensure all features are on the same scale. Models such as LASSO require the data to be standardized prior to model fitting, as some features have units that are significantly larger than other variables. This can cause issues with the regularization constraint of the LASSO problem, as it is constrained based on the magnitude of the coefficients. If standardization of the data is neglected, it can lead to the coefficients having magnitudes all over the place based on variable units. 

## Feature Selection

Feature selection for use in alternative random forest and XGBoost models was performed in 2 different ways. The first was through fitting a LASSO regression model to each horizon of the data and selecting the variables where the regression coefficients were non-zero. LASSO will push many of the variables to zero by default in order to minimize its objective function with the regularization term, so this works well for feature selection. The LASSO algorithm has a parameter, alpha, that must be tuned using cross validation. This parameter represents the regularization strength and is the coefficient to the regularization term of the objective function. Cross validation was used to find the best value of this parameter for the given data horizons. The LASSO algorithm was able to reduce the number of variables from 82 variables, down to around 60-70 features, which helped to simply the models, although the results did not necessarily improve.
The other method of feature selection that was used to reduce the dimensionality of the dataset for training the random forest and XGBoost models was principal component analysis (PCA). To find the optimal number of components to reduce the dataset down to, the explained variance of each of the top ordered principal components was plotted against the component number. After the top 8 principal components, the additional explained variance added by consecutive principal components was negligible, thus 8 principal components were computed. These top 8 principal components were used as a reduced representation of the dataset and were used to train the alternate random forest and XGBoost models in the analysis to see if this simplified dataset can improve the prediction results. 

# Regression Modeling and Hyperparameter Tuning

The regression models being used in this analysis are as follows (each model is fit for each time horizon from 1 year to 6 years):
1.	LASSO regression
2.	Ridge Regression
3.	Random Forest Regression using all original predictors (after data cleaning)
4.	XGBoost Regression using all original predictors (after data cleaning)
5.	Random Forest Regression using top 8 principal components as predictors
6.	XGBoost Regression using top 8 principal components as predictors
7.	Random Forest Regression using LASSO selected features as predictors
8.	XGBoost Regression using LASSO selected features as predictors

For each of these models, there are model specific hyperparameter that needed to be tuned to get the optimal model for prediction. These parameters were chosen with a cross validated grid search over a range of possible values, running on cloud servers to speed up this processing. The metric used to optimize these parameters was mean absolute error (MAE) and 5 folds were used in the cross validation. The reason for using MAE as the default metric, is because it is easy to interpret relative to the average value of the player, which makes intuitive sense when comparing models. On the other hand, metrics like root mean squared error (RMSE) are much more abstract because bias is being squared and square rooted, which takes away from the understanding of the error relative to the actual response. 

The parameter tuned for the LASSO and Ridge models was the alpha parameter that controls the regularization strength of the optimization problem (alpha is the coefficient of the regularization term in both cases). For the random forest model, the main parameter to tune was the number of trees in the forest. For the XGBoost model, the parameters tuned was the number of weak learners used as well as the learning rate (eta), which controls the weight of new trees added to the model. 

The reason for choosing to use these models for regression was mainly for easy interpretation. All 4 of these base models have out-of-the-box explainability of the value that each variable attribute to the final predictions through the coefficients or gini feature importance. This interpretability aspect is something not found in other models such as KNN regression and SVM regression. In addition to this interpretability, random forest and XGBoost models are known for their robust performance, as they are ensemble models that combine many weak learners together.

Once the optimal parameters were chosen for each model, performance metrics for each model were generated, and a single model was chosen to be used for prediction at each time horizon value. For each time horizon, the model used would be the one that had the best cross validated performance metrics (MAE and RMSE particularly) compared to the other models. The final model for this time horizon would be this chosen model with it’s optimized hyperparameters for that specific time horizon. This final model for each horizon was then fit to the 2021 dataset (after it had been cleaned in a similar fashion as the 2015 training set), and predictions were generated for each time horizon. The output of this process is, for each player in the 2021 dataset, a prediction of that player’s percent increase in monetary value over 1-, 2-, 3-, 4-, 5-, and 6-year periods respectively. This output can help a team manager decide which players to target based on how far out they want that player to hit their peak performance (and highest value). 

## Final Results and Evaluation

### LASSO

Shown below is the top 10 LASSO coefficients for the optimal fitted models at each time horizon. It seems that overall and potential are the most important features by a long shot, with overall having a negative correlation with the player value and the potential having a strong positive correlation with the player value. Another thing to note is that the magnitude of the overall variable coefficient increases as the horizon gets farther out, as does the potential variable in a positive direction. This makes intuitive sense, as if a player has a high overall rating in the present, it would be expected that the farther out they go from their prime in horizon years, the lower their value will become (coming down from their prime state). The same is true for the potential variable. A player with high potential is expected to increase significantly in value over the next 1-6 years, putting them in the prime years of their career. LASSO coefficients offer interpretation aspects that feature importance values cannot offer, and that is the sign of the coefficient. With the LASSO coefficients, it is obvious which variables are causing an increase in the response and which variables are causing a decrease in the response for model predictions, whereas with feature importance, all of the values are positive, so the direction of the correlation between a feature and the response is unknown.

![image](https://user-images.githubusercontent.com/55678487/118347376-7c9d9500-b508-11eb-895d-a6c6593469d9.png)

### Random Forest
Shown below are the gini feature importance values of the random forest regression model over the various time horizons. It can be observed that the age predictor is significantly the most important predictor for the random forest model, which is different than the LASSO results, but still makes intuitive sense in this case. The age of the player in the current year has a significant impact on the value of the player over a 6-year horizon.The average soccer player will achieve their peak value and performance around the 26-29 age range, with performance deteriorating after that point as the players get older. The sport of soccer has a narrow age range in which players perform at their highest level, after which, they start to deteriorate in value, so the age over a horizon period is very predictive of the player’s value on the pitch. The other top feature importance variables seem to be somewhat scattered in all areas, and tend to change out over different horizon periods, so they are likely not super important features for prediction.

![image](https://user-images.githubusercontent.com/55678487/118347384-963edc80-b508-11eb-9460-8978568e80a8.png)

### XGBoost

Shown below are the feature importance values of the XGBoost regression model over the various time horizons. Interestingly, the XGBoost feature importance values are seemingly not aligned with that of LASSO or the random forest models. The important features in this case are features that rarely showed up as important in either of the previous models. On the shorter horizon periods, defending, skill_curve, height, shooting, and heading_accuracy seem to be important features for predicting high growth players. This seems to favor goal-scorer types that can finish with their foot or head as well as solid defenders with some height to win the aerial balls, as these 2 groups of individuals are likely to be valuable players desired by the top teams in the world. On the other hand, for longer horizon periods, the skills of dribbling, shooting, penalties as well as the attributes of physic, balance, and weight_kg are most important. These features are more predictive of long-term player value, thus young players with a solid natural physique and good ball control will have high potential of increasing in value over longer spans of time in the future (4-6 years). 

![image](https://user-images.githubusercontent.com/55678487/118347397-b2db1480-b508-11eb-8c7d-1239520dc1a0.png)

### Final Model Selection

The final model to use for each time horizon was chosen using cross validated mean absolute error (MAE) and root mean squared error (RMSE). The metrics for each model were computed using 5-fold cross validation on the entire horizon dataset, using the 2015 data as the base period. For each model horizon, the XGBoost base model was chosen as the best model, as it had the best combination of MAE and RMSE across all horizon periods, which is unsurprising given its reputation as a robust algorithm through the use of boosting trees. 

### Final Player Recommendations

This final XGBoost model for each time horizon was then fit on the entire horizon dataset (with 2015 as the base year) for training. The true value-add from this analysis was to provide recommended players for team managers to scout for each time horizon based on their predicted growth in value over the horizon period. For each model, the cleaned 2021 data was passed into the XGBoost model to generate predicted response values for each player in the dataset. These predicted response values are the relative percent change in value over the time horizon for the model. The top 10 players in each time horizon with the highest predicted value growth over the horizon are listed below. For example, a 2022 maturity is a 1-year horizon from the 2021 data that was used for prediction. 
This information can be useful to team managers in the coming transfer window to help pinpoint the best purchasing decisions for their team before the next season starts and assist them in rounding out their team with emerging talent at a discounted price.

#### 2022 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2022 (%)** |
| --- | --- | --- |
| 1 | Jay Barnett | 16,251 |
| 2 | Nikolaos Baxevanos | 14,096 |
| 3 | Dannie Bulman | 13,830 |
| 4 | Hussein Omar Abdul Ghani Sulaimani | 10,587 |
| 5 | Paulo César Da Silva Barrios | 7,208 |
| 6 | 远藤保仁 | 6,257 |
| 7 | 이동국李东国 | 5,360 |
| 8 | Juan Francisco Martínez Modesto | 4,472 |
| 9 | Júlio César Cáceres López | 3,757 |
| 10 | Vitorino Hilton da Silva | 907 |

#### 2023 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2023 (%)** |
| --- | --- | --- |
| 1 | Wassim Aouachria | 16,698 |
| 2 | Ziming Liu| 16,643 |
| 3 | 张元 | 16,642 |
| 4 | 冯伯元| 16,487 |
| 5 | Mark Marleku| 16,427 |
| 6 | Yiming Yang| 16,324 |
| 7 | Díver Jesús Torres Ferrer| 16,286 |
| 8 | Gonzalo Gabriel Sánchez Franco| 16,062 |
| 9 | Shpetim Sulejmani| 15,912 |
| 10 | Kevin Raphael Diaz| 13,452 |

#### 2024 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2024 (%)** |
| --- | --- | --- |
| 1 | Kyle Hayde| 16,765 |
| 2 | Yu Zhang| 16,751 |
| 3 | Joe Woodiwiss| 16,709 |
| 4 | Uniss Kargbo| 16,603 |
| 5 | Ibrahim Bakare| 16,570 |
| 6 | Díver Jesús Torres Ferrer| 16,286 |
| 7 | Mohamed Traore| 16,225 |
| 8 | Ionuţ Radu Mitran| 16,211 |
| 9 | Jarrad Branthwaite| 13,243 |
| 10 | Jean Emile Junior Onana Onana| 7,802 |

#### 2025 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2025 (%)** |
| --- | --- | --- |
| 1 | James Akintunde| 16,838 |
| 2 | 高大伦 | 16,738 |
| 3 | Harvey Saunders | 16,630 |
| 4 | Díver Jesús Torres Ferrer | 16,286 |
| 5 | Christopher Merrie| 16,256 |
| 6 | Jay Barnett| 16,251 |
| 7 | Mohamed Traore| 16,225 |
| 8 | Isaac Hutchinson| 14,645 |
| 9 | Manuel Ugarte Ribeiro| 14,557 |
| 10 | Alfie Bates| 13,970 |

#### 2026 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2026 (%)** |
| --- | --- | --- |
| 1 | Jamie Browne| 16,832 |
| 2 | Memet Abdulla Ezmat | 16,801 |
| 3 | Luis Robles| 16,797 |
| 4 | Jiaxing Deng| 16,745 |
| 5 | Ged Garner| 16,741 |
| 6 | Christian Marcelo Silva de León | 16,645 |
| 7 | Adelin Valentin Voinescu| 16,638 |
| 8 | Nicolás Daniel Ruiz Tabárez| 16,636 |
| 9 | Diego Ferney Abadía García| 16,634 |
| 10 | Harvey Saunders| 16,630 |

#### 2027 Maturity

| **Rank** | **Player Name** | **Predicted Value Increase By 2027 (%)** |
| --- | --- | --- |
| 1 | Jamie Browne| 16,832 |
| 2 | Ricardo Dinanga| 16,831 |
| 3 | Luis Robles| 16,797 |
| 4 | Adrián Alejandro Mejía Carrasco| 16,667 |
| 5 | Saúl Alejandro Guarirapa Briceño| 16,652 |
| 6 | Harvey Saunders| 16,630 |
| 7 | Bruno Scorza Perdomo| 16,585 |
| 8 | Edgaras Dubickas| 16,481 |
| 9 | 懂洪麟| 16,479 |
| 10 | Michael Tait | 16,380 |



