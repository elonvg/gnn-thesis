import torch
import numpy as np
from deepchem.metrics import Metric
import deepchem.metrics as metrics
import pandas as pd

def train(model, train_dataset, test_dataset, loss_fn=None, epochs=100):
    print("lessgo4")
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()  # Default to MSE for regression tasks

    metric = Metric(metrics.mean_squared_error)
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = model.fit(train_dataset, nb_epoch=1)
        
        test_scores = model.evaluate(test_dataset, metrics=[metric])
        test_loss = test_scores['mean_squared_error']
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}, Test Loss = {test_loss}")
            
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    print(f"Training complete: Train Loss = {train_loss}")

    return train_losses, test_losses


def predict(model, dataset, df):
  # Function for using model to predict toxicity
  # Returns a dataframe with org and predictions

  predictions = model.predict(dataset)
  actuals = dataset.y

  # Create dataframe
  df_results = pd.DataFrame({
        'SMILES' : df['SMILES'].values,
        'Actual': actuals.flatten(),
        'Predicted': predictions.flatten()
    })

  # Convert conc back
  df_results['Actual'] = df_results['Actual'].apply(lambda x: 10**x if pd.notnull(x) else np.nan)
  df_results['Predicted'] = df_results['Predicted'].apply(lambda x: 10**x if pd.notnull(x) else np.nan)

  # Calc error
  df_results['Abs_Error'] = (df_results['Actual'] - df_results['Predicted']).abs()

  return df_results