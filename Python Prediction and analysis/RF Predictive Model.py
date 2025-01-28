import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

data = pd.read_csv(r"C:\Users\DELL\Downloads\FMCG_data.csv")

print("Data Head:")
print(data.head())
print("\nData Info:")
print(data.info())

X = data.drop(columns=['product_wg_ton'])
y = data['product_wg_ton']

categorical_columns = ['Location_type', 'zone', 'wh_owner_type', 'approved_wh_govt_certificate']
numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in categorical_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

preprocess_bar = tqdm(total=2, desc="Data Preprocessing", position=0)

X_train_processed = preprocessor.fit_transform(X_train)
preprocess_bar.update(1)

X_test_processed = preprocessor.transform(X_test)
preprocess_bar.update(1)
preprocess_bar.close()

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

rf_model = RandomForestRegressor(random_state=42)

param_combinations = [
    {'n_estimators': n, 'max_depth': max_d, 'min_samples_split': min_s, 'min_samples_leaf': min_l}
    for n in param_grid['n_estimators']
    for max_d in param_grid['max_depth']
    for min_s in param_grid['min_samples_split']
    for min_l in param_grid['min_samples_leaf']
]

grid_search_bar = tqdm(total=len(param_combinations), desc="Grid Search Progress", position=1)

best_score = float('-inf')
best_params = None

for params in param_combinations:
    rf_model.set_params(**params)
    rf_model.fit(X_train_processed, y_train)
    score = -cross_val_score(rf_model, X_train_processed, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    if score > best_score:
        best_score = score
        best_params = params
    grid_search_bar.update(1)

grid_search_bar.close()

print(f"Best Parameters: {best_params}")

rf_model.set_params(**best_params)
rf_model.fit(X_train_processed, y_train)

y_pred = rf_model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

if r2 > 0.8 and mse < 100:
    print("\nThe model is optimized with a good fit!")
else:
    print("\nThe model is not fully optimized and may require further tuning.")

save_bar = tqdm(total=1, desc="Saving Predictions", position=3)
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
file_path = r'C:\Users\DELL\Downloads\predictions.csv'
predictions_df.to_csv(file_path, index=False)
save_bar.update(1)
save_bar.close()

print(f"\nPredictions saved to '{file_path}'.")