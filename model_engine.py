import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap

# --- Data Generation (Same as before, but in its own file) ---
def generate_data(num_projects=1000):
    np.random.seed(42)
    data = {
        'project_type': np.random.choice(['Substation', 'Overhead Line', 'UG Cable'], num_projects, p=[0.4, 0.4, 0.2]),
        'terrain': np.random.choice(['Plain', 'Hilly', 'Forest'], num_projects, p=[0.5, 0.3, 0.2]),
        'budgeted_cost_crores': np.random.uniform(50, 500, num_projects),
        'vendor_performance_score': np.random.uniform(3, 10, num_projects),
        'monsoon_days': np.random.randint(30, 120, num_projects),
        'material_availability_index': np.random.uniform(4, 10, num_projects),
        'regulatory_hindrance_level': np.random.choice(['Low', 'Medium', 'High'], num_projects, p=[0.6, 0.3, 0.1]),
        'skilled_manpower_score': np.random.uniform(5, 10, num_projects)
    }
    df = pd.DataFrame(data)
    df['cost_overrun_percent'] = (5 + (10 - df['vendor_performance_score']) * 1.5 + (10 - df['material_availability_index']) * 2.0 + (df['terrain'] == 'Hilly') * 3 + (df['terrain'] == 'Forest') * 5 + np.random.normal(0, 3, num_projects)).clip(lower=0)
    df['timeline_overrun_days'] = (10 + df['monsoon_days'] * 0.4 + (df['regulatory_hindrance_level'] == 'Medium') * 20 + (df['regulatory_hindrance_level'] == 'High') * 50 + (10 - df['skilled_manpower_score']) * 4 + np.random.normal(0, 10, num_projects)).clip(lower=0)
    return df

# --- Model Training Function ---
def get_trained_models():
    df = generate_data()
    
    # Features and Targets
    X = df.drop(columns=['cost_overrun_percent', 'timeline_overrun_days'])
    y_cost = df['cost_overrun_percent']
    y_timeline = df['timeline_overrun_days']
    
    # Preprocessing
    categorical_features = ['project_type', 'terrain', 'regulatory_hindrance_level']
    numerical_features = ['budgeted_cost_crores', 'vendor_performance_score', 'monsoon_days', 'material_availability_index', 'skilled_manpower_score']
    
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    
    # Cost Model Pipeline
    model_cost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, random_state=42)
    pipeline_cost = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_cost)])
    pipeline_cost.fit(X, y_cost)
    
    # Timeline Model Pipeline
    model_timeline = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, random_state=42)
    pipeline_timeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_timeline)])
    pipeline_timeline.fit(X, y_timeline)
    
    return pipeline_cost, pipeline_timeline, X.columns.tolist()