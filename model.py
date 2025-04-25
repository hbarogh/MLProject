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
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        acc = self.accuracy(outputs.softmax(dim=-1), labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

class CombindedAIDataset(Dataset):
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
        label = 1 if "ai" in img_path.parts else 0
        return self.transform(image), torch.tensor([label], dtype=torch.long)

def build_loaders(all_imgs, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    full_dataset = CombindedAIDataset(all_imgs, transform=transform)
    train_len = int( 0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    seed = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len], generator=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)

    test_dataset = Path(list(("dataset/ImageNet100/test").rglob("*.jpeg")))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True, persistent_workers=True)

    return train_loader, val_loader
def main():
    # dataset_path = Path("dataset/ai")
    all_paths = list(Path("datasets/ai").rglob("*.png")) + \
                list((Path("datasets/ImageNet100/train").rglob("*.png")) + \
                list((Path("datasets/ImageNet100/test").rglob("*.jpeg")))
                )
    train_loader, val_loader, test_loader = build_loaders(all_paths, batch_size=32)
    model = ImageClassificationCNN()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min",
        dirpath="checkpoints/ImageClassificationCNN/",
        filename="ImageClassificationCNN_Checkpoint",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    #trainer
    trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=50, precision="16-mixed")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=val_loader)



    #
    # for wnid_folder in dataset_path.iterdir():
    #     if wnid_folder.is_dir():
    #         images = list(wnid_folder.glob('*.png'))
    #         print(f"Found {len(images)} images in folder {wnid_folder.name}")
    #
    #         train_size = int(len(images) * 0.8)
    #         val_size = len(images) - train_size
    #         seed = torch.Generator().manual_seed(42)
    #         train_images, val_images = torch.utils.data.random_split(images, [train_size, val_size], generator=seed)
    #         print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
    #         # Here you would typically load the images and labels, preprocess them, and create DataLoader instances.
    #
    #         for img_file in wnid_folder.iterdir():
    #             if img_file.suffix.lower() in ['.png']:
    #                 print(f"Processing image: {img_file.name} in folder {wnid_folder.name}")
    #
