<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['\\(','\\)'], ['$', '$']],
      displayMath: [['\\[','\\]'], ['$$', '$$']],
      <!-- skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'], -->
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


[Back to home](index.md)

# Midterm Report

## Introduction

Soccer is the most popular sport in the world, with an estimated 240 million registered players worldwide, and there is undoubtedly a lot of data present (Terrell, 2022).

Sports forecasting has been dominated by traditional statistical methods, but machine learning methods have emerged. Recent models used for sports forecasting are neural networks (Horvat & Job, 2020), such as Hubáček et. al. who use a convolutional neural network on an NBA dataset (Hubáček et al., 2019), although other researchers use regression models (e.g. linear, logistic), or support vector machines (Wilkens, 2021).

We will be using the [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer/data) by Hugo Mathien from Kaggle. We are interested in the following tables:
* __Match__:
This table contains features from over 25,000 matches played in Europe from 2008 to 2016. This dataset contains match statistics (e.g. number of cards, lineups, etc.) and betting odds from 10 providers. Betting odds will be important in our models when validating results.
* __Player_attributes__:
This table contains features such as player’s ratings in different attributes from the FIFA video game series, as well as height, weight, preferred foot, and age.
* __Team_attributes__:
This table contains features that rate teams in different aspects of the game, such as passing, dribbling, etc.

## Problem Definition

The sports betting industry is valued at 81 billion US dollars in the year 2022, and it is projected to double that by 2030. A critical piece of this industry is the spreads and odds that are bet on. For example, if the sportsbook which controls the betting odds expects Team A to beat team B by 3.5 points, if you bet on team A and they win by 4+ points then you win, else you lose, even if they win by 2 points. These spreads are essential to making money on betting, so we will use machine learning to create our own soccer game spreads, and identify which teams are overvalued and undervalued. As a first step, we will train ML models to predict match outcomes based on various feature sets.

__Disclaimer__: Online sports betting is illegal in the state of Georgia, and betting odds data are gathered from an open source database on Kaggle.


## Data Collection

We created three sets of features to train our various classifier models with:
1. Betting odds
2. Team-based attributes
3. Player-based attributes

To create these features, we had to generate features to encode match-ups between two teams. The Match data table contains betting odds from 10 providers, which already encode a match up by providing the odds of home team win, away team win, or draw For the remaining features, we computed match-up metrics using team-based and player-based ratings from the FIFA video game series. For example, for team attributes from the FIFA video game series sourced from the Team_attributes table, we computed the difference in home team metric with away team metric, and performed various aggregations (min, max, mean, median) on the features before associating them with their matches in the Match table. An example of one attribute we computed is average overall player FIFA rating per team. We then take the difference in average overall player FIFA rating from the home team and away team, and use that as the training feature.

The binary classification problem used “home team wins” as the positive label (1), and “home team draw/lose” as the negative label (0). Our dataset contained home team goals and away team goals, so we created our label accordingly – 1 if home team goals - away team goals > 0, 0 otherwise.

## Methods

Researchers have used many different machine learning algorithms to predict sports outcomes. We will use three models in this project:
1. Logistic Regression
2. Random Forest
3. Artificial Neural Network

Logistic regression models not only can be used for win/loss classification, but also give confidence in the model's decision. A random forest approach achieved an impressive 83% accuracy, and other random forest or decision tree models also outperformed regression based models (Wilkens, 2021). Lastly, artificial neural networks have shown strong results and require less preprocessing on inputs than other models (Hubáček et al., 2019).

We have trained our models on our various feature sets defined in the data collection. Each time we train the model, we perform a random sampling to generate our train dataset and test dataset. Our logistic regression models utilized a 90% train / 10% test split on the Match data table. The main data table contains ~26000 data points, so a 90% train / 10% split would result in ~23400 training records and ~2600 testing records. We handled missing values by dropping rows containing them, so our resulting training data would have less entries.

### Logistic Regression Methods
We trained various logistic regression models on three different sets of features: betting odds, match based matchup features, and player based matchup features. We used Scikit-Learn’s Logistic Regression Model instance using Newton-Cholesky and Saga solvers to optimize the parameters (scikit-learn developers, 2014).

For the betting odds and team attribute based features, the following methodology was used as a baseline:
1. Generate features
2. Perform train/test split
3. Train model on all features using training dataset
4. Evaluate feature importance
5. Determine top $k$ features based on importance which minimize test error and retrain
6. Evaluate performance and accuracy

The feature importance was evaluated using Permutation Feature Importance. The permutation feature importance metric randomly shuffles individual features one-by-one and measures the variation in the evaluation score. The features with higher variation are more important, while features with lower variation are less important (scikit-learn developers, n.d.). Additional steps were performed in some other sections.

### Random Forest Methods

## Results and Discussion

As a rule of thumb, we will target 60% accuracy for rankings, as there is a very high degree of unpredictability in sports due to the obvious _human factor_. We shoot for a “more likely than not” heuristic for our predictions, which would give the model’s user an edge in closer matchups. For example, any rational fan of the beautiful game would predict with near 100% accuracy that a 2008-2009 season Barcelona FC squad would defeat a bottom-of-the-table 2023 MLS team. ML predictions are more valuable in the contentious matchups.

### Logistic Regression Results
#### Betting Odds Features

On initial training with all 30 features (home win, away win, draw odds from 10 providers), the following results were yielded:

| __Metric__ | __Value__ |
| -- | -- |
| Train Error | 0.35403 |
| Test Error | 0.30435 |

The error was calculated by using the Jaccard similarity of the predicted labels and actual labels using the training and testing data as the accuracy metric. The Jaccard similarity was chosen because of the uniform nature of the actual labels in the binary classification problem – approximately 45% of the games in the Match dataset have a label of home_team_win = 1. The Jaccard similarity, $J$, is defined as follows for the two sets of actual and predicted labels, $Y$ and $\hat Y$ respectively:

$$J(Y, \hat Y) = \frac{|Y \cap \hat Y|}{|Y \cup \hat Y|}$$

So, the error is computed as $\texttt{error} = 1 - J(Y, \hat Y)$. Now, the following are the top 10 features by mean permutation importance in the model.

![](feature_importance_odds.png)

This clues us in to which betting odds providers give more valuable odds when it comes to predicting sports outcomes, and in a way can tell a user which provider to trust. When retraining the logistic using the top $k$ features by importance, we see the following train/test error.

![](error_analysis_odds.png)

We see that a logistic regression model performs the best when choosing the top 3 or 4 models as it minimizes the test error. Picking the top three features (SJA, B365H, VCD) we get a Jaccard similarity on testing data of 0.707, or 70.7% accuracy in predicting match outcomes! Our RMSE is 0.2935. Train/test error is fairly low, and accuracy is high, so there is good confidence in these results.

It was hypothesized that due to the similar train/test error that the model could be underfitting, so additional feature engineering was attempted through introducing polynomial features. Upon varying the highest degree, we visualized the errors:

![](error_analysis_poly_odds.png)

The lowest train and test error are curiously seen at max degree 1, so polynomial features are not the best approach. Ultimately, we settle with a 70% accuracy predictor using logistic regression trained on betting odds data!

#### Team-Based Features

The team-based FIFA attributes are separated into three categories, which have three features each:
1. Build-up Play
    1. buildUpPlaySpeed
    2. buildUpPlayDribbling
    3. buildUpPlayPassing
2. Chance Creation
    1. chanceCreationPassing
    2. chanceCreationCrossing
    3. chanceCreationShooting
3. Defence
    1. defencePressure
    2. defenceAggression
    3. defenceTeamWidth
Each of these features were then aggregated into mean, median, min, and max values per team from the Team_attributes table, as each team has multiple entries in the table. So, we end up with 36 attribute features.

##### Feature Selection and Training by Importance

Initial training on all team-based attribute features yielded the following errors on training and testing data:

| __Metric__ | __Value__ |
| -- | -- |
| Train Error | 0.36822 |
| Test Error | 0.44886 |

This results in a Jaccard index accuracy on the test data of 0.5511, or around 55%, which is lower than the betting odds data. We again evaluate the top ten features by permutation importance and we get:

![](feature_importance_attr.png)

We see that attributes relating to defense and chance creation are much more valuable than build up speed. Performing the same error analysis by varying the top “k” features used in training yields the following plot:

![](error_analysis_attr.png)

There is much greater fluctuation in test error, although train error decreases sharply (by around 0.08) as more features are chosen. By retraining the model on the top ~25 features which minimizes test error, we get a slight decrease in test error (new error = 0.3625 from old error = 0.3682), a decrease of only 1.55%, which is not significant.

##### By-Category Feature Selection and Training

A more manual feature-selection approach was used to improve the accuracy of the model. Manually training the feature on each individual category yielded the following:

| __Feature Category__ | __Metric__ | __Value__ |
| -- | -- | -- |
| Build-up Play | RMSE on test | 0.42613636363636365 |
| | Accuracy on test | 0.5738636363636364 |
| Chance Creation | RMSE on test | 0.4090909090909091 |
| | Accuracy on test | 0.5909090909090909 |
| Defense | RMSE on test | 0.5113636363636364 |
| | Accuracy on test | 0.48863636363636365 |

Chance creation seems to be the top performing category with a 59% accuracy, a good 4% increase from the previous accuracy score. However, we can do better using the important feature selection metrics. Sorting the chance creation aggregations by importance yields the following:

![](feature_importance_chance_creation.png)

The error fluctuates with respect to the top number of features chosen as follows:

![](error_analysis_chance_creation.png)

Test error is very clearly minimized using the top 6 features. Retraining the model using the top 6 most important features results in a Jaccard similarity of 0.6134, which tells us we have a 61% accurate logistic regression predictor based solely on the FIFA video game series team aggregate chance creation metrics! We are happy here because our 61% accuracy is sufficient based on our heuristics.

Here is a visualization plotting the top two features against the probability of the match being a home team win:

![](chance_creation_prob.png)

We see that higher lower probabilities (darker colors) tend towards the left end while higher probabilities (lighter colors) tend towards the right.


### Random Forest Results


### Conclusions
To conclude, we see the following sufficiently accurate (>60% accuracy) models:
* 70% accurate logistic regression trained on 25 betting odds features
* 61% accurate logistic regression trained on 6 FIFA team chance creation features

## [Timeline](https://gtvault-my.sharepoint.com/:x:/g/personal/sajjan3_gatech_edu/EZLBLWmNKIlOhDSoWb220_8B6iRV6UzX8bXvDjJ6bf01vA?e=dXrga7)
See link above

## Contributors

| Name | Contribution |
| -- | -- |
| Daniel | Introduction, Logistic Regression (betting odds and team based features)|
| Elijah | Problem Definition, Random Forest|
| Sabina | Timeline, slides for video, Random Forest|
| Xander | Potential Results and Discussion, Logistic Regression (player based features) |
| Matthew | Methods, Random Forest|

## References
Horvat, T., & Job, J. (2020). The use of machine learning in sport outcome prediction: A review. WIREs Data Mining and Knowledge Discovery, 10(5). https://doi.org/10.1002/widm.1380

Hubáček, O., Šourek, G., & Železný, F. (2019). Exploiting sports-betting market using machine learning. International Journal of Forecasting, 35(2), 783–796. https://doi.org/10.1016/j.ijforecast.2019.01.001

scikit-learn developers. (n.d.). 4.2. Permutation feature importance — scikit-learn 0.23.1 documentation. Scikit-Learn.org. https://scikit-learn.org/stable/modules/permutation_importance.html

scikit-learn developers. (2014). sklearn.linear_model.LogisticRegression — scikit-learn 0.21.2 documentation. Scikit-Learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Terrell, E. (2022). Research Guides: Sports Industry: A Research Guide: Soccer. Guides.loc.gov. https://guides.loc.gov/sports-industry/soccer

Wilkens, S. (2021). Sports prediction and betting models in the machine learning age: The case of tennis. Journal of Sports Analytics, 7(2), 1–19. https://doi.org/10.3233/jsa-200463
