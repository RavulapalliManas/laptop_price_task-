import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl 
import pandas as pd

#Loading the cleaned data
data = pd.read_csv("/Users/avi/Desktop/university/sem_5/Machine_Learning/Avi_Dhall_A1/laptop_price_task-/data/cleaned_data.csv")
# data = pd.read_csv("/Users/manasvenkatasairavulapalli/Desktop/Computer Science stuff/Pure CS/Introduction to Machine Learning/Assignments/life-expectancy-prediction/data/cleaned_data.csv")

#isolating the target variable and vectorizing it for matrix operations
y = data['Price']
y = np.array(y).reshape(-1,1)


#Isolating the features and getting the number rows and columns for future use 
X = data.drop(columns=['Price'])
X = np.c_[np.ones(X.shape[0]), X]
m,n = X.shape


class LinearRegression:
    def __init__(self, regularization= "none", method = "none"):
        self.method = method
        self.regularization = regularization
        self.theta = None 
        
    def linear_regression_model(self, X, y, regularization= "none", method = "none"): 
        #Since Matrix operations are faster in numpy better to convert them       
        X = np.array(X)
        y = np.array(y)
        
        #intializing theta with small random values
        theta = np.random.rand(X.shape[1],1) * 0.01
        Lambda = 0.1
        alpha = 0.01
        iterations = 1000
        m = X.shape[0]
        
        #Normal Equation 
        if method == "normal_equation":
            if regularization == "L1":
                raise ValueError("L1 regularization not supported with normal equation")
            
            elif regularization == "L2":
                I = np.eye(X.shape[1])
                I[0,0] = 0
                self.theta = np.linalg.inv(X.T @ X + Lambda * I) @ X.T @ y
            
            else:   
                self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
            
        #Gradient Descent 
        elif method == "gradient_descent":
            for i in range(iterations):
                
                #precomputing predicted y's and error for easier interpretability of the dradient descent formula 
                y_hat = X @ theta 
                e = y_hat - y
                gradient_J = (1/m) * (X.T @ e)
    
                #The original formula before derivation is J = (1/2m) * sum((y_hat - y)^2) + (Lambda/m) * sum(|theta|)
                if regularization == "L1":
                    regurlarization_term = (Lambda/m) * np.sign(theta)
                    regurlarization_term[0] = 0
                    gradient_J += regurlarization_term
                
                #original formula before derivation is J = (1/2m) * sum((y_hat - y)^2) + (Lambda/2m) * sum(theta^2) 
                elif regularization == "L2":
                    regurlarization_term = (Lambda/m) * theta
                    regurlarization_term[0] = 0
                    gradient_J += regurlarization_term
                #no regularization
                elif regularization == "none":
                    pass 
                else:
                    return ValueError("Invalid regularization type")
                
                #Our update rule originally written as Theta_j+1 = Theta_j - alpha * dJ/dTheta_j 
                theta = theta - alpha * gradient_J
            
            self.theta = theta
        else:
            raise ValueError("Invalid method type")
    
    #Training and testing the models performance using K fold cross validation. 
    def KCV(self, X, y, k, regularization= "none", method = "none"):
        #Splitting the data into k buckets using the class kfoldcv
        buckets = kfoldcv.splitting(X, k)
        
        #metrics per fold
        Performance = []
        
        #Predictions per fold
        Predictions = []
        
        Average_Performance = {"avg_mse": None, "avg_rmse": None, "avg_r2": None}  # Default in case no folds
        # We define the training and test data and iterate till we have trained and tested on all the buckets
        for n,(train_indices, test_indices) in enumerate(buckets):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            self.linear_regression_model(X_train, y_train, regularization, method) 
            
            y_hat = self.Predict(X_test)
            
            #adding predicted values per bucket
            Predictions.append({"Bucket": n, "y_test": y_test, "y_hat": y_hat})

            #calculating metrics per bucket
            mse_val = Metrics.mse(y_test, y_hat)
            rmse_val = Metrics.rmse(y_test, y_hat)
            r2_val = Metrics.r2_score(y_test, y_hat)
            
            #Storing performance
            Performance.append({"Bucket": n, "MSE": mse_val, "RMSE": rmse_val, "R2": r2_val})
            Average_Performance = {"avg_mse": np.mean([m["MSE"] for m in Performance]),
            "avg_rmse": np.mean([m["RMSE"] for m in Performance]),
            "avg_r2": np.mean([m["R2"] for m in Performance])}

        return Average_Performance, Predictions, Performance


    def Predict(self,X):
        X = np.array(X)
        #y_hat is the predicted value 
        return X @ self.theta
    
    #save model
    def save(self, file):
        with open(file, 'wb') as f:
            pkl.dump(self, f)
    
    #load model
    def load(self,file):
        with open(file, 'rb') as f:
            model = pkl.load(f)
        return model

class kfoldcv:
    def __init__(self, splits = 0):
        self.splits = splits
    
    #define a splitting function to create the buckets and train and test data (Source GeeksforGeeks)
    @staticmethod
    def splitting(data, k):
        bucket_size = len(data) // k
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        buckets = []
        for i in range(k):
            start = i * bucket_size
            end = start + bucket_size
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            buckets.append((train_indices, test_indices))
        return buckets

   
            
        
class Metrics:

    #mean squared error function
    @staticmethod
    def mse(y, y_hat):
        return np.mean((y - y_hat) ** 2)
    
    #root mean squared error function
    @staticmethod
    def rmse(y, y_hat):
        return np.sqrt(Metrics.mse(y, y_hat))

    #R2 score function 
    @staticmethod
    def r2_score(y, y_hat):       
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        return 1 - (ss_res / ss_total)
    
def main():
    model = LinearRegression(regularization="L2", method="normal_equation")

    # run training with k-bucket cross-validation
    avg_perf, predictions, perf_per_bucket = model.KCV(
    X, y, k=5, regularization="L2", method="normal_equation"
    )

    print("Average Performance:", avg_perf)  # will now print avg_mse, avg_rmse, avg_r2


if __name__ == "__main__":
    print(">>> Script started successfully <<<")
    main()