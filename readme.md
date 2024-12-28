# NBA Data Analysis Documentation

## Overview
This project involves building predictive models and performing exploratory data analysis (EDA) on NBA data to evaluate team and player performance. The workflow integrates machine learning techniques, statistical modeling, and feature engineering, with a focus on Elo ratings, mutual information, and PCA for feature selection.

---

## Key Components

### 1. Data Preprocessing and Parsing
- **File Location**: `data/scores`
- **Description**: Parses HTML box score files using `BeautifulSoup` and `pandas`.
- **Features Extracted**:
  - Line scores
  - Basic and advanced statistics
  - Team standings
  - Game outcomes (win/loss)
- **Output**: `team_stats.csv`

#### Example Code:
```python
soup = parse_html(box_score)
line_score = read_line_score(soup)
standings = read_scores(soup)
summary = read_stats(soup, team, "basic")
```

---

### 2. Feature Engineering
#### **Elo Ratings**
- Initializes and updates Elo ratings for teams.
- Accounts for home advantage and seasonal reset with weighted initial Elo.
- Adds Elo features (`Elo_Team`, `Elo_Team.1`) to the dataset.

#### Example:
```python
elo = initialize_elo(teams, initial_elo=1500)
data = add_elo_scores(df)
```

#### **Mutual Information**
- Calculates mutual information scores to rank predictors based on their predictive power for the target variable.
- Focuses on the top 20 features for modeling.

#### Example:
```python
mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': predictors, 'mi': mi}).sort_values('mi', ascending=False)
```

---

### 3. Machine Learning Models

#### **Models Used**:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **Voting Classifier**

#### **Backtesting Framework**
- Validates models across multiple NBA seasons.
- Performs calibration using `CalibratedClassifierCV` for improved probability estimates.

#### Example Backtest:
```python
def backtest(data, predictors, model, start=3, step=1):
    for i in range(start, len(seasons), step):
        model.fit(train[predictors], train['Home-Team-Win'])
        predictions, predictions_prob = get_predictions(model, test[predictors])
```

#### **Elo-Based Predictions**
- Elo ratings are directly used to predict game outcomes based on relative team strength.

---

### 4. Dimensionality Reduction
#### **Principal Component Analysis (PCA)**
- Reduces the dimensionality of one-hot encoded team data.
- Identifies the optimal number of components to retain 95% variance.

#### Example:
```python
pca = PCA(n_components=225)
pca.fit(transformed_train_df)
num_components, cumulative_variance = find_optimal_components(pca.explained_variance_ratio_)
```

---

### 5. Visualization and Evaluation

#### **Calibration Curve**
Plots predicted probabilities against actual probabilities to evaluate model calibration.

#### **ROC Curve**
Displays the tradeoff between true positive rate (TPR) and false positive rate (FPR).

#### **Confusion Matrix**
Visualizes classification performance by showing true positives, false positives, true negatives, and false negatives.

#### Examples:
```python
plot_calibration(prob_df)
plot_roc_curve(prob_df)
plot_confusion_matrix(test, predictions)
```

---

### 6. Performance Metrics
- **Accuracy**
- **F1 Score**
- **AUC (Area Under Curve)**

#### Example:
```python
accuracy = round(accuracy_score(test['Home-Team-Win'], predictions), 4)
f1 = round(f1_score(test['Home-Team-Win'], predictions), 4)
auc = round(roc_auc_score(test['Home-Team-Win'], predictions), 4)
```

---

### 7. Notable Findings
- Elo-based predictions achieved consistent accuracy across seasons.
- Mutual information revealed key predictors like team standings and advanced stats.
- PCA identified optimal feature sets, retaining 95% variance with reduced complexity.
- Calibration curves show the model's predicted probabilities align well with actual probabilities, especially in higher probability ranges.
- ROC curve analysis yielded an AUC of 0.69, indicating moderate model performance.
- Accuracy, F1, and AUC metrics for each season demonstrate year-over-year stability:

| Season | Accuracy | F1    | AUC   |
|--------|----------|-------|-------|
| 2016   | 0.6432   | 0.7170| 0.6168|
| 2017   | 0.6551   | 0.7273| 0.6281|
| 2018   | 0.6582   | 0.7233| 0.6364|
| 2019   | 0.6597   | 0.7206| 0.6446|
| 2020   | 0.6195   | 0.6933| 0.6027|
| 2021   | 0.6435   | 0.7065| 0.6288|
| 2022   | 0.6222   | 0.6878| 0.6041|
| 2023   | 0.6494   | 0.7106| 0.6349|

---

## Future Directions
- Explore additional ensemble techniques beyond soft voting.
- Implement neural network-based models for player-level prediction.
- Integrate real-time data updates for in-season predictions.

---

## File Outputs
- `team_stats.csv`: Processed game-level data.
- `prob_df`: Probabilities for each prediction.
- `acc_df`: Accuracy metrics for each season.

