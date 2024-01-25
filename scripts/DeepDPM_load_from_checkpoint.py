import argparse
import torch
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
from src.datasets import CustomDataset
import pytorch_lightning as pl
import os


# LOAD MODEL FROM CHECKPOINT
cp_path ="D:/研究生课程/前沿技术/DeepDPM-main/saved_models/USPS/USPS_exp/epoch=299-step=21899.ckpt"# E.g.: "./saved_models/MNIST_N2D/default_exp/epoch=57-step=31725.ckpt"
cp_state = torch.load(cp_path)
data_dim = 10 # E.g. for MNIST, it would be 10 if the network was trained on the embeedings supplied, or 28*28 otherwise.
K = cp_state['state_dict']['cluster_net.class_fc2.weight'].shape[0] 
hyper_param = cp_state['hyper_parameters']
args = argparse.Namespace()
for key, value in hyper_param.items():
    setattr(args, key, value)

model = ClusterNetModel.load_from_checkpoint(
    checkpoint_path=cp_path,
    input_dim=data_dim,
    init_k = K,
    hparams=args
    )

# Example for inference :
model.eval()
dataset_obj = CustomDataset(args)
train_loader, val_loader = dataset_obj.get_loaders()
cluster_assignments = []
for data, label in val_loader:
    soft_assign = model(data)
    hard_assign = soft_assign.argmax(-1)
    cluster_assignments.append(hard_assign)
print(cluster_assignments)
# Inside your training loop or script
trainer = pl.Trainer(callbacks=[pl.callbacks.ProgressBar()])
trainer.fit(model, train_loader, val_loader)
# Accessing the loss values
train_loss = trainer.callback_metrics['train_loss'].numpy()
val_loss = trainer.callback_metrics['val_loss'].numpy()

# Plotting the loss
import matplotlib.pyplot as plt

plt.plot(train_loss, label='Training Loss',color='red')
plt.plot(val_loss, label='Validation Loss',color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


