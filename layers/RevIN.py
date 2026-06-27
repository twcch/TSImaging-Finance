import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # 計算局部序列的均值和變異數，維度: (Batch, 1, Features)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # 反正規化時自動對齊輸出通道數：當 c_out < num_features（如 MS 模式
        # enc_in=4、c_out=1）時，取前 c_out 個通道的統計量，與既有基準模型
        # stdev[:, 0, :c_out] 的切片慣例一致；c_out == num_features 時為無作用。
        n = x.shape[-1]
        if self.affine:
            x = x - self.affine_bias[:n]
            x = x / (self.affine_weight[:n] + self.eps*self.eps)
        x = x * self.stdev[..., :n]
        x = x + self.mean[..., :n]
        return x