import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from project.datasets import BraTS2019, RandomCrop, RandomRotFlip, ToTensor
from project.models import net_factory_3d
from project.utils.losses import DiceLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/BraTS2019")
    parser.add_argument("--exp", type=str, default="thesis_supervised_3d")
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs") / args.exp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model definition happens here.
    model = net_factory_3d("unet_3d", in_chns=1, class_num=2).to(device)

    dataset = BraTS2019(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(2)

    iteration = 0
    # Training starts here.
    for _ in tqdm(range(args.max_iterations // len(loader) + 1), desc="Epoch", ncols=80):
        for batch in loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            pred = model(image)
            pred_soft = torch.softmax(pred, dim=1)

            # Loss functions are calculated here.
            loss = 0.5 * (ce_loss(pred, label) + dice_loss(pred_soft, label.unsqueeze(1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                logging.info("iter=%d loss=%.4f", iteration, loss.item())
            if iteration >= args.max_iterations:
                torch.save(model.state_dict(), run_dir / "last_model.pth")
                return


def main():
    args = parse_args()
    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    train(args)


if __name__ == "__main__":
    main()
