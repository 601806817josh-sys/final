import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from project.datasets import BraTS2019, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from project.models import net_factory_3d
from project.utils import losses, ramps
from project.utils.eval_3d import test_all_case


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/BraTS2019")
    parser.add_argument("--exp", type=str, default="thesis_mean_teacher_3d")
    parser.add_argument("--model", type=str, default="unet_3d")
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--labeled_num", type=int, default=25)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def get_current_consistency_weight(iter_num, consistency, rampup):
    return consistency * ramps.sigmoid_rampup(iter_num // 150, rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = Path("runs") / f"{args.exp}_{args.labeled_num}_{args.model}"
    snapshot_path.mkdir(parents=True, exist_ok=True)

    # Model definition happens here (student + EMA teacher).
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=2).to(device)
    ema_model = net_factory_3d(net_type=args.model, in_chns=1, class_num=2).to(device)
    for param in ema_model.parameters():
        param.detach_()

    db_train = BraTS2019(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()]),
    )
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(str(snapshot_path / "log"))
    iter_num, best_performance = 0, 0.0
    max_epoch = args.max_iterations // len(trainloader) + 1

    # Training starts here (main epoch/batch loops).
    for _ in tqdm(range(max_epoch), ncols=80, desc="Epoch"):
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].to(device)
            label_batch = sampled_batch["label"].to(device)

            unlabeled_volume_batch = volume_batch[args.labeled_bs :]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(unlabeled_volume_batch + noise)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            # Loss functions are calculated here.
            loss_ce = ce_loss(outputs[: args.labeled_bs], label_batch[: args.labeled_bs])
            loss_dice = dice_loss(outputs_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_ce + loss_dice)
            consistency_weight = get_current_consistency_weight(iter_num, args.consistency, args.consistency_rampup)
            consistency_loss = torch.mean((outputs_soft[args.labeled_bs :] - ema_output_soft) ** 2)
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("train/loss", loss.item(), iter_num)
            writer.add_scalar("train/lr", lr_, iter_num)

            if iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size)
                dice = avg_metric[:, 0].mean()
                if dice > best_performance:
                    best_performance = dice
                    torch.save(model.state_dict(), snapshot_path / "best_model.pth")
                logging.info("iter=%d, val_dice=%.4f", iter_num, dice)
                model.train()

            if iter_num >= args.max_iterations:
                writer.close()
                return


def main():
    args = parse_args()
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = args.deterministic

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    train(args)


if __name__ == "__main__":
    main()
