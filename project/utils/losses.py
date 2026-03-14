import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = [(input_tensor == i * torch.ones_like(input_tensor)) for i in range(self.n_classes)]
        return torch.cat(tensor_list, dim=1).float()

    @staticmethod
    def _dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        return 1 - (2 * intersect + smooth) / (torch.sum(score * score) + torch.sum(target * target) + smooth)

    def forward(self, inputs, target):
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), "predict & target shape do not match"
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes
