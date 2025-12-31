import pandas as pd
import numpy as np

def load_model_predictions(csv_file):
    """
    Load model predictions from a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file containing model predictions.
    
    Returns:
    pd.DataFrame: DataFrame containing the model predictions.
    """
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} model predictions from {csv_file}")
    print(df.head())
    return df

def load_real_coordinates(csv_file):
    """
    Load real coordinates from a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file containing real coordinates.
    
    Returns:
    pd.DataFrame: DataFrame containing the real coordinates.
    """
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} real coordinates from {csv_file}")
    print(df.head())
    
    #compress it to combine all test_types into one row per test_type with lists of latitudes and longitudes, labeled as latitude1, longitude1, latitude2, longitude2, etc.
    compressed_df = df.groupby('test_type').apply(lambda x: pd.Series({
        **{f'real_latitude{i+1}': lat for i, lat in enumerate(x['latitude'].values)},
        **{f'real_longitude{i+1}': lon for i, lon in enumerate(x['longitude'].values)}
    })).reset_index()
    print("Compressed real coordinates:")
    print(compressed_df.head())
    return compressed_df

def main():
    model_predictions_file = 'Roy/ML/Second_Level_ML/predicted_coordinates_all_testtypes.csv'
    real_coordinates_file = 'Roy/ML/Second_Level_ML/real_coordinates.csv'
    
    model_predictions_df = load_model_predictions(model_predictions_file)
    real_coordinates_df = load_real_coordinates(real_coordinates_file)
    
    # add the real coordinates to the model predictions dataframe
    merged_df = pd.merge(model_predictions_df, real_coordinates_df, left_on='test_type', right_on='test_type')
    print(merged_df.head())
    merged_df.to_csv('Roy/ML/Second_Level_ML/merged_model_predictions_real_coordinates.csv', index=False)

if __name__ == "__main__":
    main()