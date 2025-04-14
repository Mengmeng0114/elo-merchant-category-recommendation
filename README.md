# Elo Merchant Category Recommendation

This project is based on the Kaggle competition [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation). The goal is to predict customer loyalty scores for merchants based on their transaction history.

## Project Overview

Elo, Brazil's largest payment brand, has built machine learning models to understand customer preferences. This project aims to predict customer loyalty scores for merchants based on their transaction history.

### Dataset

The dataset is available on Kaggle and includes:
- `train.csv` - Training data with target values
- `test.csv` - Test data for predictions
- `merchants.csv` - Merchant information
- `new_merchant_transactions.csv` - New merchant transaction data
- `historical_transactions.csv` - Historical transaction data

## Project Structure

The project is organized into several Jupyter notebooks:

1. `elo_data_processing.ipynb` - Data preprocessing and cleaning
2. `elo_feature_engineering.ipynb` - Feature creation and transformation
3. `elo_randomforest.ipynb` - Random Forest model implementation
4. `elo_lightgbm_tpe.ipynb` - LightGBM model with Tree-structured Parzen Estimators (TPE)
5. `elo_xgboost.ipynb` - XGBoost model implementation
6. `kaggle_elo_test.ipynb` - Final testing and submission

## Models Implemented

The project implements three different approaches:
1. Random Forest
2. LightGBM with Tree-structured Parzen Estimators (TPE)
3. XGBoost with NLP Features
   - Utilizes CountVectorizer and TF-IDF for text feature processing
   - Processes merchant IDs, category IDs, and location information as text features
   - Combines traditional numerical features with NLP-derived features
   - Implements Bayesian optimization for hyperparameter tuning

## Getting Started

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/elo-merchant-category-recommendation/data)
2. Place the data files in the `elo-merchant-category-recommendation` directory
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebooks in sequence:
   - Start with data processing
   - Follow with feature engineering
   - Then run the model notebooks
   - Finally, use the test notebook for predictions

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- jupyter

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for hosting the competition
- Elo for providing the dataset
- All participants who shared their insights in the competition discussion 