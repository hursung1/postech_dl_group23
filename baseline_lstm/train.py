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
        
        if len(x.size()) == 4: # CharNgram
            x.squeeze_(2)

        out, last_hidden = model(x)
        loss = crit(out, y)
        loss.backward()
        optim.step()

        pred = torch.argmax(out, dim=1)
        correct += torch.sum((pred == y))
        total += len(y)

        # # add adversarial noise
        # grad = [] 
        # for param in model.parameters():
        #     print(param.grad.shape)
        #     grad.append(param.grad)

        # optim.zero_grad()
        # last_hidden += grad
        # loss = crit(last_hidden, y)
        # loss.backward()
        # optim.step()

        pbar.set_description(f"EPOCH {epoch+1} Training Loss: {loss.item():.4f} ACC: {correct/total:.4f}")


def eval(epoch, model, dataloader, device, is_test):
    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    
    if is_test:
        predictions = None
        for x in pbar:
            x = x.to(device)

            if len(x.size()) == 4: # CharNgram
                x.squeeze_(2)

            out = model(x)
            pred = torch.argmax(out, dim=1)
            # print(pred.shape)
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
        
        return predictions.detach().cpu().tolist()

    else:
        total = 0
        correct = 0
        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            if len(x.size()) == 4: # CharNgram
                x.squeeze_(2)

            out = model(x)
            pred = torch.argmax(out, dim=1)

            correct += torch.sum((pred == y))
            total += len(y)
        
        acc = correct / total
        print(f"Validation Acc: {acc:.4f}")
        return acc