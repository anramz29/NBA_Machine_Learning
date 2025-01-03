# NBA Data Analysis Documentation

## Overview
This project involves building predictive models and performing exploratory data analysis (EDA) on NBA data to evaluate team and player performance. The workflow integrates machine learning techniques, statistical modeling, and feature engineering, with a focus on Elo ratings, mutual information, and PCA for feature selection.



1. **Cross-validation with PCA**: Applies dimensionality reduction using PCA to retain 95% of variance and evaluates a Random Forest classifier.
2. **Elo Rating Calculation**: Tracks team strength over time using a custom Elo system without data leakage.
3. **Historical Statistics Calculation**: Computes win percentages and game counts for teams up to the current game.

---

## **Code Structure**


### **1. Elo Rating Calculation**
Tracks team Elo ratings chronologically without looking ahead, ensuring no data leakage.

- **Steps**:
  1. Initialize all team ratings to a base value (e.g., 1500).
  2. Adjust team ratings after each game based on the result and the expected score.
  3. Carry over 75% of the previous season's Elo rating to the next season, if applicable.

- **Key Parameters**:
  - `initial_elo`: Starting Elo rating for new teams (default: 1500).
  - `k`: Sensitivity factor for Elo updates (default: 20).
  - `home_advantage`: Elo boost for the home team (default: 100).

- **Key Outputs**:
  - `Elo_Team`: Elo rating of the home team before the game.
  - `Elo_Team.1`: Elo rating of the away team before the game.

- **Code Snippet**:
  ```python
  def calculate_elo_chronologically(data, initial_elo=1500, k=20, home_advantage=100):
      for idx, row in data.iterrows():
          home_team = row['TEAM_NAME']
          away_team = row['TEAM_NAME.1']
          
          home_elo = team_elos.get(home_team, initial_elo)
          away_elo = team_elos.get(away_team, initial_elo)
          
          home_expected = 1 / (1 + 10 ** (-(home_elo - away_elo + home_advantage) / 400))
          home_win = row['Target']
          
          team_elos[home_team] += k * (home_win - home_expected)
          team_elos[away_team] += k * ((1 - home_win) - (1 - home_expected))
  ```

---

### **2. Historical Statistics Calculation**
Calculates win percentages and game counts for each team up to the current game without data leakage.

- **Steps**:
  1. Track team statistics (wins, losses, and total games) for each season.
  2. For each game, store the historical win percentage for both teams before updating their stats.

- **Key Outputs**:
  - `home_win_pct`: Home team's win percentage before the game.
  - `away_win_pct`: Away team's win percentage before the game.
  - `total_games`: Total games played by the home team before the game.

- **Code Snippet**:
  ```python
  def calculate_historical_stats(data):
      for idx, row in data.iterrows():
          home_team = row['TEAM_NAME']
          away_team = row['TEAM_NAME.1']
          
          home_stats = season_stats[season][home_team]
          away_stats = season_stats[season][away_team]
          
          data.at[idx, 'home_win_pct'] = home_stats['wins'] / max(home_stats['games'], 1)
          data.at[idx, 'away_win_pct'] = away_stats['wins'] / max(away_stats['games'], 1)
          data.at[idx, 'total_games'] = home_stats['games']
  ```

---

### **3. Full Data Processing**
Combines all steps into a single function to process the dataset without data leakage.

- **Steps**:
  1. Sort the dataset chronologically by game date.
  2. Apply Elo rating calculation.
  3. Compute historical statistics.

- **Key Outputs**:
  - Processed dataset with Elo ratings and historical stats.

- **Code Snippet**:
  ```python
  def process_data_without_leakage(data):
      data = data.sort_values('Date').copy()
      data = calculate_elo_chronologically(data)
      data = calculate_historical_stats(data)
      return data

  data_processed = process_data_without_leakage(df)
  ```




### **4. Cross-validation with PCA**
This step applies Principal Component Analysis (PCA) and evaluates the model's performance in a cross-validation framework.

- **Steps**:
  1. Split the data into training and validation sets using K-Fold cross-validation.
  2. Apply PCA to retain 95% of variance.
  3. Train a Random Forest classifier on the PCA-reduced data.
  4. Evaluate validation accuracy and store PCA feature importance.

- **Key Outputs**:
  - Validation scores (`cv_scores`)
  - Number of components for 95% variance (`cv_n_components`)
  - Feature importance from PCA (`cv_feature_importance`)

- **Code Snippet**:
  ```python
  for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X_scaled)), total=n_splits, desc="Cross-validation"):
      X_train = X_scaled[train_idx]
      X_val = X_scaled[val_idx]
      y_train = target.iloc[train_idx]
      y_val = target.iloc[val_idx]
      
      pca = PCA(n_components=0.95)
      X_train_pca = pca.fit_transform(X_train)
      X_val_pca = pca.transform(X_val)
      
      clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
      clf.fit(X_train_pca, y_train)
      val_score = clf.score(X_val_pca, y_val)
  ```

<img src="images/PCA.png" alt="Scree Plot" width="900">


---

### 5. Machine Learning Models

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

#### Original Features
##### Performance Metrics:

- Higher accuracy (0.639) and AUC (0.619) suggest better ranking and classification ability when using original features.
- The slightly lower F1 score (0.707) compared to PCA features indicates that the original features might struggle slightly more with balanced predictions in scenarios where false positives and false negatives carry significant weight.
Calibration:

- The calibration curve is smoother and closely aligned with the diagonal, indicating that predicted probabilities closely match actual probabilities.

- This reflects the model's reliability when using the full feature set, potentially due to a richer representation of underlying relationships.

##### Calibration Curve
<img src="images/Original_Calibration_Curve.png" alt="Calibration Curve" width="700">


#### PCA-Reduced Features
##### Performance Metrics:

- Accuracy (0.619): A slight decrease compared to the original features, likely because PCA removes some variance associated with predictive but less dominant features.
- F1 Score (0.715): The increase suggests better handling of balanced prediction tasks, such as when classes are imbalanced or costs of misclassification differ.
- AUC (0.584): A decrease indicates reduced ability to rank predictions by confidence, possibly due to the dimensionality reduction discarding some subtle but predictive relationships.
Calibration:

While slightly noisier, the curve still aligns reasonably well with the diagonal, indicating that PCA retains enough variance to make meaningful probabilistic predictions.
However, the noisiness could reflect information loss or weaker correlations between reduced features and the target variable.

##### Calibration Curve


<img src="images/PCA_Calabration_Curve.png" alt="Calibration Curve" width="700">



## Observations:

Calibration Curves:

The PCA features show a slightly noisier calibration but still align reasonably well with the diagonal.
This suggests the model generalizes decently, even after dimensionality reduction.
Metrics Context:

AUC being low (0.584) suggests weaker ranking ability, but the F1 score is strong, meaning PCA features might work better for balanced prediction tasks rather than nuanced ranking.

---

Here’s the updated documentation with additional steps detailing your future plans to use a neural network and explore other feature selection techniques like Sequential Feature Selection (SFS):

---

## NBA Data Analysis Documentation

### 7. Future Steps

#### **1. Using a Neural Network for Prediction**
To further improve predictive performance, especially in capturing complex nonlinear relationships, I plan to implement a neural network model.

- **Architecture**:
  - Input layer: Features from PCA or other feature selection techniques.
  - Hidden layers: Use 2–3 dense layers with ReLU activation to capture nonlinearities.
  - Output layer: Single neuron with sigmoid activation for binary classification (Home Team Win or Loss).

- **Steps**:
  1. Use TensorFlow/Keras or PyTorch for implementation.
  2. Experiment with different optimizers (e.g., Adam) and learning rates.
  3. Perform hyperparameter tuning for:
     - Number of hidden layers and neurons.
     - Batch size and learning rate.
     - Dropout rates to prevent overfitting.
  4. Use cross-validation for robust evaluation.

- **Expected Benefits**:
  - Improved ability to capture complex feature interactions.
  - Better calibration of predicted probabilities through techniques like temperature scaling.

- **Code Snippet**:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout

  def build_neural_network(input_dim):
      model = Sequential([
          Dense(128, activation='relu', input_dim=input_dim),
          Dropout(0.3),
          Dense(64, activation='relu'),
          Dropout(0.3),
          Dense(1, activation='sigmoid')
      ])
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      return model
  ```

---

#### **2. Exploring Sequential Feature Selection (SFS)**
To identify the most informative features, I will experiment with **Sequential Feature Selection** (SFS). This technique adds or removes features iteratively to optimize model performance.

- **Steps**:
  1. Use scikit-learn's `SequentialFeatureSelector` with models like Logistic Regression or Random Forest as the estimator.
  2. Perform both **forward selection** (starting with an empty set of features) and **backward elimination** (starting with all features).
  3. Compare the selected features with those identified by PCA and mutual information.

- **Metrics for Evaluation**:
  - Accuracy, F1 Score, and AUC for the selected feature subset.
  - Computational efficiency compared to PCA.

- **Code Snippet**:
  ```python
  from sklearn.feature_selection import SequentialFeatureSelector
  from sklearn.ensemble import RandomForestClassifier

  def perform_sfs(X, y):
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      sfs = SequentialFeatureSelector(clf, n_features_to_select=10, direction='forward', cv=5)
      sfs.fit(X, y)
      selected_features = sfs.get_support(indices=True)
      return selected_features
  ```

- **Expected Benefits**:
  - Identifies specific features that contribute most to predictive performance.
  - Can complement PCA by selecting interpretable features in the original feature space.

---


