# Milk Quality Classification with Random Forest

## Description

This project demonstrates a machine learning pipeline for classifying milk quality grades using a Random Forest Classifier in Python. The code shows how to:

1. Upload a CSV dataset directly into Google Colab.
2. Perform exploratory data analysis and basic cleaning.
3. Scale numerical features (`Temperature` and `pH`).
4. Split the data into training and test sets.
5. Train a Random Forest model and evaluate its performance.

## File Structure

```
├── milknew.csv        # Raw dataset file (uploaded via Colab)
├── notebook.ipynb     # Google Colab notebook with code
├── README.md          # Project description and instructions
```

## Code Overview

1. **Data Upload** (Method 1 - `files.upload()`):

   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

   Prompts the user to select and upload `milknew.csv` from their local machine.

2. **Data Loading and Inspection**:

   ```python
   import io, pandas as pd
   milk = pd.read_csv(io.StringIO(uploaded['milknew.csv'].decode('utf-8')))
   print(milk.head(), milk.info(), milk.isnull().sum())
   ```

   Reads the uploaded CSV into a DataFrame and prints initial records, data types, and missing-value counts.

3. **Preprocessing**:

   - **Column Rename**: Fixes a spelling error in the header from `Temprature` to `Temperature`.
   - **Scaling**: Applies `StandardScaler` to `Temperature` and `pH` for mean=0, variance=1.

4. **Train/Test Split**:

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.31, random_state=42
   )
   ```

   Splits features and target into training (69%) and testing (31%) subsets.

5. **Model Training**:

   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(
       n_estimators=150, max_depth=20, bootstrap=True, random_state=42
   )
   model.fit(X_train, y_train)
   ```

   Trains a Random Forest with 150 trees, max depth of 20, using bootstrap sampling.

6. **Evaluation**:

   - **Confusion Matrix** and **Classification Report** to inspect precision, recall, and F1-score per class.
   - **Accuracy Score** to measure overall performance.
   - **Overfitting Check** comparing training vs. test accuracy.

## Random Forest Algorithm

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees. Key strengths:

- **Robustness**: Reduces overfitting by averaging multiple trees.
- **Nonparametric**: Makes no strong assumptions about the data distribution.
- **Feature importance**: Provides estimates of feature relevance.

### How it works:

1. **Bootstrap Sampling**: Creates multiple subsets of the original data by sampling with replacement.
2. **Tree Construction**: For each subset, grows a decision tree by selecting a random subset of features at each split.
3. **Aggregation**: For classification, each tree votes for a class; the forest selects the class with the majority vote.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- Google Colab (for `files.upload()` and Drive mounting)

## Usage

1. Open the `notebook.ipynb` in Google Colab.
2. Run the upload cell and select `milknew.csv`.
3. Execute all cells to train and evaluate the model.
4. Review results and feature importances.

