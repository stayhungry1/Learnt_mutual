import torch.optim as optim
import torch.nn as nn
import math
import torch

def mse2psnr(mse):
    # 根据Hyper论文中的内容，将MSE->psnr(db)
    # return 10*math.log10(255*255/mse)
    return 10 * math.log10(1 / mse)

def configure_optimizers(net, learning_rate, aux_learning_rate):
# def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        # lr=args.learning_rate,
        lr=learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        # lr=args.aux_learning_rate,
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer

class RateDistortionLoss(nn.Module): #只注释掉了109行的bpp_loss, 08021808又加上了
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    # def forward(self, output, target, lq, x_l, x_enh): #0.001
    def forward(self, output, target):  # 0.001 #, lq, x_l, x_enh
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # # out["mse_loss"] = self.mse(lq, target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] #lambda越小 bpp越小 越模糊 sigma预测的越准，熵越小
        # out["loss"] = self.mse(x_l, x_enh)
        # out["loss"] = self.mse(x_l, x_enh) + out["mse_loss"]
        return out
