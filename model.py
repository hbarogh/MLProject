import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from torchmetrics.classification import Accuracy
from lightning.pytorch.callbacks import ModelCheckpoint


class ImageClassificationCNN(L.LightningModule):
    def __init__(self):
        super(ImageClassificationCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.accuracy = Accuracy(task="binary", num_classes=2)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, labels.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, labels.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        acc = self.accuracy(preds, labels.int())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class CombinedAIDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 1.0 if "datasets/ai" in img_path.as_posix() else 0.0

        return image, torch.tensor([label], dtype=torch.float32)

def gather_paths(split: str):
    real_root = Path("datasets/ImageNet100") / split
    real_imgs = list(real_root.glob("*.jpeg"))
    ai_root = Path("datasets/ai")
    ai_imgs = list(ai_root.rglob("*.png"))
    return real_imgs + ai_imgs

def make_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    def make_loader(split):
        paths = gather_paths(split)
        dataset = CombinedAIDataset(paths, transform=transform)
        shuffle = True if split == "train" else False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
    train_loader = make_loader("train")
    val_loader = make_loader("val")
    test_loader = make_loader("test")
    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = make_loaders(batch_size=32)
    model = ImageClassificationCNN()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min",
        dirpath="checkpoints/ImageClassificationCNN/",
        filename="ImageClassificationCNN_Checkpoint_Best",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    # trainer
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=50,
        precision="16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=val_loader)

if __name__ == "__main__":
    main()
