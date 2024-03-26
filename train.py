import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy
import lightning as L

class MyModel(L.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.linear = nn.LazyLinear(10)
        self.metric = Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        accuracy = self.metric(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, batch_size=32)


model = MyModel()
trainer = L.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader)
