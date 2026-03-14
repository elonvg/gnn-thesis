import torch
import numpy as np
from deepchem.metrics import Metric
import deepchem.metrics as metrics

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