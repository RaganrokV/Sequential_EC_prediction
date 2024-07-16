import torch
from torch import nn
import time



def train_model(model, train_loader, epochs, lr, batch_size, train_size):
    best_val_loss = float("inf")
    best_model = None
    loss_func = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)

    train_loss_all = []
    model.train()  # Turn on the train mode
    total_loss = 0.

    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(train_loader):

            time_start = time.time()

            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pre_y = model(x)

            loss = loss_func(pre_y, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  #梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

            time_end = time.time()
            time_c= (time_end - time_start)*100

            total_loss += loss.item()
            log_interval = int(train_size / batch_size / 5)
            if (step + 1) % log_interval == 0 and (step + 1) > 0:
                cur_loss = total_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.6f} | '
                      'loss {:5.5f} | time {:8.2f}'.format(
                    epoch, (step + 1), train_size // batch_size, scheduler.get_lr()[0],
                    cur_loss, time_c))
                total_loss = 0

        if (epoch + 1) % 5 == 0:
            print('-' * 89)
            print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
            print('-' * 89)
            train_loss_all.append(train_loss / train_num)

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model = model

        scheduler.step()

    return best_model

