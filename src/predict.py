import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from train_model import Regression, Metrics
from data_preprocessing import preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--predictions_output_path', type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.data_path)

    cleaned_data = preprocess(data)
    
    # Separate features and target variable
    X = cleaned_data.drop(columns=['Price'])
    y = cleaned_data['Price'].values.reshape(-1, 1)
    X = np.array(X)
    y = np.array(y)
    
    # Load the trained model
    with open(args.model_path, 'rb') as f:
        model = pkl.load(f)
    
    # Make predictions
    predictions = X @ model.theta
    predictions = predictions.flatten()
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({'prediction': predictions})
    predictions_df.to_csv(args.predictions_output_path, index=False)
 
    # Calculate and save metrics
    metrics = Metrics(y, predictions)
    mse = metrics.mean_squared_error()
    rmse = metrics.root_mean_squared_error()
    r2 = metrics.r2_score()
    with open(args.metrics_output_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"R2 Score: {r2}\n")

