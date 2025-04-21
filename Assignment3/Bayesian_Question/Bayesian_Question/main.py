#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
import time

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    # pass
    X_train = pd.read_csv("train_data.csv")
    X_val = pd.read_csv("validation_data.csv")
    return X_train, X_val

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # pass
    edges = [
        ('End_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Route_Type'),
        ('End_Stop_ID', 'Distance'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Route_Type', 'Fare_Category'),
        ('Start_Stop_ID', 'Zones_Crossed'),
        ('Start_Stop_ID', 'Distance'),
        ('End_Stop_ID', 'Fare_Category'),
        ('Distance', 'Fare_Category'),
        ('End_Stop_ID', 'Route_Type'),
        ('Start_Stop_ID', 'Fare_Category')
    ]

    graph = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(graph, df)
    bn.plot(graph)

    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    # pass
    edges = [
        ('Distance', 'Fare_Category'),
        ('Start_Stop_ID', 'End_Stop_ID'),
        ('Route_Type', 'Fare_Category'),
        ('Start_Stop_ID', 'Fare_Category'),
        ('Zones_Crossed', 'Fare_Category'),
        ('Distance', 'Zones_Crossed'),
    ]

    graph = bn.make_DAG(edges)
    model = bn.parameter_learning.fit(graph, df)
    bn.plot(model)

    return model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    # pass
    graph = bn.structure_learning.fit(df, methodtype='hc')
    model = bn.parameter_learning.fit(graph, df)
    bn.plot(graph)

    return model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    # pass
    try:
        with open(fname, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved {fname}")
    except Exception as e:
        print(f"Error saving model {fname}: {e}")

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    # start = time.time()
    base_model = make_network(train_df.copy())
    # end = time.time()
    save_model("base_model.pkl", base_model)
    # print("Time of Execution: ", end-start)

    # Create and save pruned model
    # start = time.time()
    pruned_network = make_pruned_network(train_df.copy())
    # end = time.time()
    save_model("pruned_model.pkl", pruned_network)
    # print("Time of Execution: ", end-start)

    # Create and save optimized model
    # start = time.time()
    optimized_network = make_optimized_network(train_df.copy())
    # end = time.time()
    save_model("optimized_model.pkl", optimized_network)
    # print("Time of Execution: ", end-start)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

