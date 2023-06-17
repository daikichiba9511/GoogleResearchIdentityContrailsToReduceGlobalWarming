import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=68, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, x):
        return self.mlp(x)


def main():
    net = Net().cuda()
    optimizer = optim.Adam(net.parameters())
    for step in range(3):
        feature = torch.randn((30, 68)).cuda()
        target = torch.randn((30, 1)).cuda()

        output = net(feature)
        print(output.shape)

        loss: torch.Tensor = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()

        print(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]["lr"]
        print(lr)


if __name__ == "__main__":
    main()
