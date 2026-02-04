"""
ETD Deviation Model - Prediction Functions
Separate module for ETD deviation model loading and predictions.
"""
import pickle
import pandas as pd


def load_deviation_model(path: str = "artifacts/etd_deviation_model.pkl"):
    """Load the ETD deviation prediction model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_features_for_deviation(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Build feature dataframe for ETD deviation prediction.
    
    Args:
        df: Input dataframe with raw data
        model: Trained pipeline model
        
    Returns:
        Feature dataframe ready for prediction
    """
    df = df.copy()
    
    # Get expected features from the model pipeline
    preprocessor = model.named_steps["pre"]
    num_cols = preprocessor.transformers[0][2]  # numeric columns
    
    # Ensure all expected columns exist
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    # Return only the features the model expects
    return df[num_cols]


def predict_etd_deviation(df: pd.DataFrame, model) -> pd.Series:
    """
    Predict ETD deviation (days between estimated and actual departure).
    
    Args:
        df: DataFrame with shipment data
        model: Trained ETD deviation model
        
    Returns:
        Series with predicted ETD deviation in days
    """
    X = build_features_for_deviation(df, model)
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="predicted_etd_deviation_days")
