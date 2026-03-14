import torch

def train(model, train_dataset, test_dataset, loss_fn=None, epochs=100):

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()  # Default to MSE for regression tasks

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.fit(train_dataset, nb_epoch=1)
        train_loss = get_loss(model, train_dataset, loss_fn)
        train_losses.append(train_loss)

        test_loss = get_loss(model, test_dataset, loss_fn)
        test_losses.append(test_loss)
        print("lessgo")
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}, Test Loss = {test_loss}")
            
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    print(f"Training complete: Train Loss = {train_loss}")

    return train_losses, test_losses

def get_loss(model, dataset, loss_fn=None):
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()  # Default to MSE for regression tasks

    X = torch.tensor(dataset.X, dtype=torch.float32)
    y = torch.tensor(dataset.y, dtype=torch.float32)

    model.model.eval()
    with torch.no_grad():
        y_pred = model.model(dataset)
        loss = loss_fn(y_pred, y)

    return loss.item()