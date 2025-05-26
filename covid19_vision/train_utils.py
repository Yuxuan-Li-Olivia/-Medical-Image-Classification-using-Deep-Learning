import torch
from torch import nn, optim

def train_model(net, train_loader, val_loader, num_epochs, lr, device):
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(num_epochs):
        net.train()
        running_loss, running_correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (y_hat.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        net.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                val_correct += (y_hat.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}")

    return train_losses, train_accs, val_accs