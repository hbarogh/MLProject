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
import random

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
#
# def gather_paths(split: str):
#     if split == "train":
#         real_dirs = list(Path("datasets/ImageNet100").glob("train.*"))
#     elif split == "val":
#         real_dirs = Path("datasets/ImageNet100/val.X")
#     else:
#         real_dirs = Path("datasets/ImageNet100/test.X")
#     real_imgs = []
#     for dir in real_dirs:
#         real_imgs.extend(list(dir.glob("*.jpeg")))
#
#
#     ai_root = Path("datasets/ai")
#     ai_imgs = list(ai_root.rglob("*.png"))
#     return real_imgs, ai_imgs

def _all_jpegs(path: Path):
    "Collect jpg/jpeg/JPEG under a directory tree"
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
    test_dir   = root / "val.X"        # rename on-disk unnecessary

    # grab every real image once
    real_train = sum((_all_jpegs(d) for d in train_dirs), start=[])
    real_test  = _all_jpegs(test_dir)

    # deterministic shuffle so train/val don’t overlap
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(real_train), generator=g).tolist()

    split_point = int(0.15 * len(real_train))   # 15 %
    val_indices = set(idx[:split_point])

    real_val   = [real_train[i] for i in val_indices]
    real_train = [real_train[i] for i in idx[split_point:]]

    if split == "train":
        real_paths = real_train
    elif split == "val":
        real_paths = real_val
    else:  # test
        real_paths = real_test

    # ---------- FAKE IMAGES ----------
    fake_all   = list(Path("datasets/ai").rglob("*.png"))
    k = len(fake_all)
    # use the SAME indexing trick so the three splits don't overlap
    fake_idx = torch.randperm(k, generator=g).tolist()

    if split == "train":
        fake_subset = [fake_all[i] for i in fake_idx]        # 100 %
    elif split == "val":
        fake_subset = [fake_all[i] for i in fake_idx[: k // 6]]   # ~15 %
    else:
        fake_subset = [fake_all[i] for i in fake_idx[k // 6 : k // 3]]  # next 15 %

    return real_paths, fake_subset



def make_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    def make_loader(split):
        real_paths, ai_paths = gather_paths(split)
        # if split == "train":
        #     fake_subset = ai_paths[:]
        # elif split == "val":
        #     fake_subset = ai_paths[:len(ai_paths)//5] # gives us 20% of the data
        # else:
        #     fake_subset = ai_paths[len(ai_paths)//5: len(ai_paths) // 3]
        if split == "train":
            n = len(ai_paths)
            real_paths = random.sample(real_paths, n)
            print(f"Train: {len(real_paths)} real images, {n} fake images")
        paths = real_paths + ai_paths
        random.shuffle(paths)
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
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.benchmark = True
    train_loader, val_loader, test_loader = make_loaders(batch_size=32)
    print(f"Train len: {len(train_loader.dataset)}")
    print(f"Val len:   {len(val_loader.dataset)}")
    print(f"Test len:  {len(test_loader.dataset)}")
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
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
