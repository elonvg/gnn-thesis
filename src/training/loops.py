import copy

import torch

from .metrics import regression_metrics


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze()
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()

            predictions.append(out.cpu())
            targets.append(batch.y.cpu())

    avg_loss = total_loss / len(loader)
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    rmse, mae = regression_metrics(predictions, targets)

    return avg_loss, rmse, mae


def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs=100, device="cpu"):
    model = model.to(device)

    best_test_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "test_loss": [], "test_rmse": [], "test_mae": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_rmse"].append(test_rmse)
        history["test_mae"].append(test_mae)

        if scheduler is not None:
            scheduler.step(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    return model, history
