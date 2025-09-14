import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from train_model import Regression, Metrics
from data_preprocessing import pre_process

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--predictions_output_path', type=str, required=True)
    args = parser.parse_args()
 
    # Load dataset
    data = pd.read_csv(args.data_path)

    cleaned_data = pre_process(data)
    
    # Separate features and target variable
    X = cleaned_data.drop(columns=['Life expectancy'])
    y = cleaned_data['Life expectancy'].values.reshape(-1, 1)
    X = np.array(X)
    y = np.array(y)
    
    
    # Load the trained model
    with open(args.model_path, 'rb') as f:
        model = pkl.load(f)
    
    # Make predictions
    predictions = model.Predict(X).flatten()
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({'prediction': predictions})
    predictions_df.to_csv(args.predictions_output_path, index=False)
    
    print("y sample:", y[:5].flatten())
    print("predictions sample:", predictions[:5])

 
    mse = Metrics.mse(y, predictions)
    rmse = Metrics.rmse(y, predictions)
    r2 = Metrics.r2_score(y, predictions)

    with open(args.metrics_output_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Root Mean Squared Error: {rmse}\n")
        f.write(f"R2 Score: {r2}\n")
        
if __name__ == "__main__":
    main()