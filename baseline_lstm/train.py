import torch
from tqdm import tqdm

def trainer(epoch, model, dataloader, crit, optim, device):
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    total = 0
    correct = 0
    for (x, y) in pbar:
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        optim.step()

        pred = torch.argmax(out, dim=1)
        correct += torch.sum((pred == y))
        total += len(y)

        pbar.set_description(f"EPOCH {epoch+1} Training Loss: {loss.item():.4f} ACC: {correct/total:.4f}")


def eval(epoch, model, dataloader, device):
    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    total = 0
    correct = 0
    for (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        pred = torch.argmax(out, dim=1)
        correct += torch.sum((pred == y))
        total += len(y)

        # pbar.set_description(f"EPOCH {epoch+1} Validation ACC: {correct/total:.4f}")
    acc = correct/total
    print(f"Validation Acc: {acc:.4f}")
    return acc