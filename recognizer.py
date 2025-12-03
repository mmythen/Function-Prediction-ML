import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas


class numsDataset(Dataset):
    def __init__(self, path):
        df = pandas.read_csv(path)


        self.x = torch.tensor(df[["num1", "num2"]].values, dtype=torch.float32)
        #regression target
        self.y = torch.tensor(df["result"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
class numsModel(nn.Module):
    def __init__(self):
        super().__init__()
        #non-linear regress
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)


dataset = numsDataset("nums.csv")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = numsModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(200):
    for x, y in loader:
        #optimize loop
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch: {epoch}, loss: {loss}\n")




model.eval()
with torch.no_grad():
    sample = torch.tensor([[50.0, 200.0]])
    res_pred = model(sample)
    print("predicted result:", res_pred.item())



