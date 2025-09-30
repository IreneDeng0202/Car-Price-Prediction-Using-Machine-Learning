## 1. 📌Project Overview
This project predicts used car selling_price (regression) from vehicle features.



## 2.📊 Dataset
- **Target**: selling_price (continuous)
- **Independent features**: vehicle_age, km_driven, mileage, engine, max_power, seats, model (high-cardinality), seller_type, fuel_type, transmission_type



## 3. ⚙️Methodology
- **Preprocessing**: numeric imputation (median), categorical imputation + OneHot, target-encoding for high-cardinality (model) inside a ColumnTransformer.
- **Pipeline**: Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler()), ('reg', estimator)]) (use 'reg' as step name).
- **Models tuned**: RandomForestRegressor, GradientBoostingRegressor, XGBRegressor.
- **Hyperparameter search**: RandomizedSearchCV (e.g. n_iter=30, cv=5) with multi-metric scoring {"R2":"r2","MAE":"neg_mean_absolute_error","MSE":"neg_mean_squared_error"} and refit='R2'.



## 4. 📈Results
Top tuned models: RF, GB, XGB.
Final chosen model: XGBoost — Test R² = 0.9479.
Include diagnostic plots: Actual vs Predicted , Residual distribution, Residual vs Predicted. 
