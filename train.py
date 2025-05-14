import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HMDataset
from model import TemporalFusionTransformer
from utils.model_utils import save_checkpoint
import yaml

def train(config_path: str = 'configs/base_config.yaml'):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize dataset and loader
    dataset = HMDataset('./', seq_len=config['seq_len'], pred_len=config['pred_len'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    # Model and optimizer
    model = TemporalFusionTransformer(num_features=4)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.HuberLoss()

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}')

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=f'checkpoint_{epoch}.pth')

        print(f'Epoch {epoch} Average Loss: {total_loss/len(loader):.4f}')

if __name__ == '__main__':
    train()
