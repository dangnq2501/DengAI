"""CPU PyTorch approximation of the canonical SELU/RMSprop training recipe.

Preserves architecture, loss, batches per epoch and dropout; initialization,
shuffle implementation and optimizer epsilon semantics differ from TensorFlow.
"""
import numpy as np
import torch
from torch import nn


def fit(x, y, predict_x, city, config, seed, epochs, steps, checkpoints):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], config.hidden_1), nn.SELU(),
        nn.Dropout(config.dropout_1), nn.Linear(config.hidden_1, config.hidden_2),
        nn.SELU(), nn.Dropout(config.dropout_2), nn.Linear(config.hidden_2, 1))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate,
                                   alpha=.9, eps=1e-7)
    x, y = torch.as_tensor(x), torch.as_tensor(y).reshape(-1, 1)
    predict_x = torch.as_tensor(predict_x)
    order = np.arange(len(y))
    cursor = len(order)
    snapshots, history = [], []
    best, wait = -float('inf'), 0
    for epoch in range(1, epochs + 1):
        total = 0.
        model.train()
        for _ in range(steps):
            if cursor >= len(order):
                if city == 'sj':
                    rng.shuffle(order)
                cursor = 0
            indices = order[cursor:cursor + 16]
            cursor += 16
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[indices]) - y[indices]).abs().mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        mae = total / steps
        # Preserve the historical mode=max callback, including its unusual direction.
        if mae > best + 1e-4:
            best, wait = mae, 0
        else:
            wait += 1
            if wait >= (3 if city == 'sj' else 5):
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * .8, 1e-6)
                wait = 0
        history.append({'epoch': epoch, 'mae': mae, 'learning_rate': optimizer.param_groups[0]['lr']})
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                snapshots.append(model(predict_x).numpy().reshape(-1))
    return np.stack(snapshots), history, model
