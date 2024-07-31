from torch.utils.data import DataLoader
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from dataset import SegDataset
from GroupUnet3d import GroupUnet3d
from Unet3d import Unet


# torch.set_float32_matmul_precision('medium')


train_dataset = SegDataset("./data/train/images", "./data/train/masks")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

val_dataset = SegDataset("./data/val/images", "./data/val/masks")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


model = GroupUnet3d()


checkpoint_callback = ModelCheckpoint(
    dirpath="models",
    filename="GroupUnetSmall_{epoch}",
    every_n_epochs=5
)


logger = TensorBoardLogger("tb_logs", name="New_GroupUnet_Small_OneBatch")

trainer = Trainer(logger=logger, max_epochs=1000, accelerator="gpu", overfit_batches=1, callbacks=checkpoint_callback)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# model = Unet()

# checkpoint_callback = ModelCheckpoint(
#     dirpath="models",
#     filename="NormalUnetSmall_{epoch}",
#     every_n_epochs=5,
# )

# logger = TensorBoardLogger("tb_logs", name="NormalUnetSmall")

# trainer = Trainer(logger=logger, max_epochs=100, accelerator="gpu", precision="bf16-mixed", callbacks=checkpoint_callback)
# trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


