import torch
import numpy as np
from deepchem.metrics import Metric
import deepchem.metrics as metrics
import pandas as pd
import copy


def train_dc(model, train_dataset, test_dataset, loss_fn=None, epochs=100):
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

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze() # remove extra dim -> shape: batch_size
        loss = loss_fn(out, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)

    return avg_loss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze() # remove extra dim -> shape: batch_size
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()

            predictions.append(out.cpu())
            targets.append(batch.y.cpu())
    
    avg_loss = total_loss / len(loader)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item() # Root Mean Squared Error - sensitive to large errors
    mae = torch.abs(predictions - targets).mean().item() # Mean Absolute Error - more interpretable, less sensitive to outliers

    return avg_loss, rmse, mae

def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs=100, device='cpu'):
    
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_rmse, val_mae = evaluate(model, test_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_state) # Load the best model state after training

    return model, history

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