import torch
from torch.nn import BCELoss, MSELoss
from torch import Tensor


class AsymmetricFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma = gamma  # balancing between classes
        self.zeta = zeta  # balancing between active/inactive frames

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        losses = -(
            ((1 - y_hat) ** self.gamma) * y * torch.clamp_min(torch.log(y_hat), -100)
            + (y_hat**self.zeta) * (1 - y) * torch.clamp_min(torch.log(1 - y_hat), -100)
        )

        return torch.mean(losses)


class F1Loss(torch.nn.modules.loss._Loss):
    def __init__(self) -> None:
        super(F1Loss, self).__init__()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        tp = (y_hat * y).sum(0)
        fp = (y_hat * (1 - y)).sum(0)
        fn = ((1 - y_hat) * y).sum(0)
        soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = cost.mean()  # average on all labels

        return macro_cost


class DoubleF1Loss(torch.nn.modules.loss._Loss):
    def __init__(self) -> None:
        super(DoubleF1Loss, self).__init__()

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.
        This version uses the computation of soft-F1 for both positive and negative class for each label.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        tp = (y_hat * y).sum(0)
        fp = (y_hat * (1 - y)).sum(0)
        fn = ((1 - y_hat) * y).sum(0)
        tn = ((1 - y_hat) * (1 - y)).sum(0)
        soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
        cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
        macro_cost = cost.mean()  # average on all labels
        return macro_cost
