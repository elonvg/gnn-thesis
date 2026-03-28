import torch
import numpy as np
from deepchem.metrics import Metric
import deepchem.metrics as metrics
import pandas as pd
import copy

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze() # remove extra dim -> shape: batch_size
        loss = loss_fn(out, batch.y)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            pred_denorm = out #* target_std + target_mean
            target_denorm = batch.y #* target_std + target_mean

            predictions.append(pred_denorm.cpu())
            targets.append(target_denorm.cpu())
    
    avg_loss = total_loss / len(loader)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item() # Root Mean Squared Error - sensitive to large errors
    mae = torch.abs(predictions - targets).mean().item() # Mean Absolute Error - more interpretable, less sensitive to outliers

    return avg_loss, rmse, mae

def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs=100, device='cpu'):
    
    model = model.to(device)

    best_test_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'test_loss': [], 'test_rmse': [], 'test_mae': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_rmse'].append(test_rmse)
        history['test_mae'].append(test_mae)

        scheduler.step(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
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