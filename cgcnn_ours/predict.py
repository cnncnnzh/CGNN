# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""


def evaluate(
    model,
    data,
    loss_func,
    device,
):
    total_loss = 0
    count = 0
    for data in loader:
        data = data.to(device=device)
        pred = model(data)
        loss = loss_func(torch.squeeze(pred), torch.squeeze(data.y))
        total_loss += loss.item()
        # count += pred.size(0)
        count += 1
    return total_loss/count


def predict(predict_options):
    """
    Use pre-trained model to predict new data and save the results
    """
    















if __name__ == '__main__':
    predict_options = {'model':
        }