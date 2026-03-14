import torch

def train(model, train_dataset, test_dataset, loss_fn=None, epochs=100):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = model.fit(train_dataset, nb_epoch=1)
        train_losses.append(train_loss)

        test_loss = model.evaluate(test_dataset, metrics=[], per_task_metrics=False)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}")
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    print(f"Training complete: Train Loss = {train_loss}")

    return train_losses, test_losses