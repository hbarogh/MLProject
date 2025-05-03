import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from torchmetrics.classification import Accuracy, BinaryF1Score, BinaryConfusionMatrix
from lightning.pytorch.callbacks import ModelCheckpoint
import random


class ImageClassificationCNN(L.LightningModule):
    def __init__(self, pos_weight=None):
        super(ImageClassificationCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.f1Score = BinaryF1Score(threshold=0.5)
        self.confusionMatrix = BinaryConfusionMatrix(threshold=0.5)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.4).int()
        acc = self.accuracy(preds, labels.int())
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.4).int()
        acc = self.accuracy(preds, labels.int())
        loss = self.criterion(outputs, labels)
        f1_score = self.f1Score(preds, labels.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("f1_score", f1_score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.4).int()
        acc = self.accuracy(preds, labels.int())
        loss = self.criterion(outputs, labels)
        f1_score = self.f1Score(preds, labels.int())
        self.confusionMatrix(preds, labels.int())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("f1_score", f1_score, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        cm = self.confusionMatrix.compute()
        print("\nConfusion matrix (test):\n", cm.cpu().numpy())
        self.confusionMatrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
                }


class TransferResNet(L.LightningModule):
    def __init__(self, pos_weight=None, freeze_backbone=True):
        super().__init__()
        # Load pretrained ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove old head
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.accuracy = Accuracy(task="binary")
        self.f1Score = BinaryF1Score(threshold=0.5)
        self.confusionMatrix = BinaryConfusionMatrix(threshold=0.5)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", self.accuracy(preds, y.int()), prog_bar=True)
        self.log(f"{stage}_f1", self.f1Score(preds, y.int()), prog_bar=True)
        if stage == "test":
            self.confusionMatrix(preds, y.int())
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def on_test_epoch_end(self):
        print("Confusion matrix (test):", self.confusionMatrix.compute().cpu().numpy())
        self.confusionMatrix.reset()

    def configure_optimizers(self):
        lr = 1e-3 if any(p.requires_grad for p in self.backbone.parameters()) else 1e-4
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

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


def get_all_paths(path: Path):
    """
    Collect jpg/jpeg/JPEG under a directory tree
    """
    return list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.JPEG"))


def gather_paths(split: str):
    """
    split ∈ {'train','val','test'}
      • train → all train.X1-X4 images minus the 15 % reserved for val
      • val   → that 15 % subset taken from the train.X* folders
      • test  → the original val.X folder supplied by ImageNet-100
    Returns (real_paths, fake_paths)
    """
    # ---------- REAL IMAGES ----------
    root = Path("datasets/ImageNet100")

    train_dirs = [root / f"train.X{i}" for i in range(1, 5)]
    test_dir = root / "val.X"  # rename on-disk unnecessary

    # grab every real image once
    real_train = sum((get_all_paths(d) for d in train_dirs), start=[])
    real_test = get_all_paths(test_dir)

    # deterministic shuffle so train/val don’t overlap
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(real_train), generator=g).tolist()

    split_point = int(0.15 * len(real_train))  # 15 %
    val_indices = set(idx[:split_point])

    real_val = [real_train[i] for i in val_indices]
    real_train = [real_train[i] for i in idx[split_point:]]

    if split == "train":
        real_paths = real_train
    elif split == "val":
        real_paths = real_val
    else:  # test
        real_paths = real_test

    # ---------- FAKE IMAGES ----------
    fake_all = list(Path("datasets/ai").rglob("*.png"))
    random.seed(42)
    random.shuffle(fake_all)
    k = len(fake_all)
    train_split = int(0.6 * k)  # 60 %
    val_split = int(0.8 * k)  # 20 %


    if split == "train":
        fake_subset = fake_all[:train_split]  # 60 %
    elif split == "val":
        fake_subset = fake_all[train_split:val_split]  # ~20 %
    else:
        fake_subset = fake_all[val_split:]  # next 20 %
    return real_paths, fake_subset


def make_loaders(batch_size=32):
    #transforms for the CNN model
    # train_transforms = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])
    #
    # val_transforms = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])

    #transforms for the transfer learning model
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def make_loader(split):
        real_paths, ai_paths = gather_paths(split)
        if split == "train":
            n_ai = len(ai_paths)
            print(f"Train: {len(real_paths)} real images, {n_ai} fake images")
            real_paths = random.sample(real_paths, len(ai_paths) * 2)
            paths = real_paths + ai_paths
            n_real = len(real_paths)
            print(f"Train: {n_real} real images, {n_ai} fake images inside Make loaders function")
            weights = [1.0 if "datasets/ai" in p.as_posix() else n_ai / n_real for p in paths]
            sampler = WeightedRandomSampler(weights, num_samples=len(paths), replacement=True)
            transform = train_transforms
            pos_weight = torch.tensor([n_real / n_ai], dtype=torch.float32)
            dataset = CombinedAIDataset(paths, transform=transform)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler= sampler,
                num_workers=8,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4
            )
            return loader, pos_weight
        else:
            real_paths = random.sample(real_paths, len(ai_paths))
            paths = real_paths + ai_paths
            transform = val_transforms
            dataset = CombinedAIDataset(paths, transform=transform)
            print(f"Val: {len(real_paths)} real images, {len(ai_paths)} fake images")
            print(f"Test: {len(real_paths)} real images, {len(ai_paths)} fake images")
            shuffle = False
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4

            )
            return loader

    train_loader, pos_weight = make_loader("train")
    val_loader = make_loader("val")
    test_loader = make_loader("test")
    return train_loader, val_loader, test_loader, pos_weight


def main():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.benchmark = True
    train_loader, val_loader, test_loader, pos_weight = make_loaders(batch_size=32)
    print(f"Train len: {len(train_loader.dataset)}")
    print(f"Val len:   {len(val_loader.dataset)}")
    print(f"Test len:  {len(test_loader.dataset)}")

    #regular CNN section that we had before:
    # model = ImageClassificationCNN(pos_weight=pos_weight.to("cuda"))
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss", mode="min",
    #     dirpath="checkpoints/ImageClassificationCNN/",
    #     filename="ImageClassificationCNN_Checkpoint_Best",
    #     save_top_k=1
    # )

    #this is the transfer learning section that I am testing out as well:
    model = TransferResNet(pos_weight=pos_weight.to("cuda"), freeze_backbone=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min",
        dirpath="checkpoints/TransferResNet/",
        filename="TransferResNet_Checkpoint_Best",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    # trainer
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=30,
        precision="16-mixed",
        accelerator="gpu" if torch.cuda.is_available() else "cpu"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
