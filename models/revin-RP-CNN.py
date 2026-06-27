import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    RP-CNN + RevIN（對應 RP-CNN 的 RevIN 版本）。

    結構與 models/RP-CNN.py 完全相同（連續型 Recurrence Plot 影像化 + 簡單
    2D-CNN），差異在於預測（forecast）路徑加入可學習仿射的 RevIN（Reversible
    Instance Normalization）：在影像化之前對輸入做 'norm'、在輸出端做 'denorm'，
    使預測值還原回原始尺度，緩解金融序列的非平穩性。

    分類（classification）路徑維持與 RP-CNN 基準一致：RP 內部已對每個視窗做
    z-score 標準化，且分類輸出為 logits 無需反正規化，故不額外加入 RevIN。

    註：RevIN 自行處理正規化，故與 run.py 的 --use_norm 旗標無關。

    Config 對應：
        enc_in   -> 影像通道數（輸入變數數，亦為 RevIN 通道數）
        dropout  -> 分類頭 dropout
        c_out    -> 預測任務輸出變數數
        num_class-> 分類類別數（由 Exp_Classification 動態注入）
        rp_gamma -> RP 距離敏感度（可選，預設 1.0）
    """

    C1, C2 = 16, 32  # 兩層卷積的通道數（簡單影像模型）

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.seq_len if self.task_name == 'classification' else configs.pred_len
        self.c_out = configs.c_out
        self.gamma = getattr(configs, 'rp_gamma', 1.0)   # 距離敏感度

        # RevIN 實例正規化（可學習仿射），用於 forecast 的輸入/輸出尺度還原
        self.revin_layer = RevIN(configs.enc_in)

        # 2D-CNN 特徵萃取（兩層卷積 + 最大池化）
        self.features = nn.Sequential(
            nn.Conv2d(configs.enc_in, self.C1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.C1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(self.C1, self.C2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.C2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        if self.task_name == 'classification':
            # 自適應池化到固定網格，避免 FC 維度依賴 seq_len
            self.gap = nn.AdaptiveAvgPool2d((4, 4))
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.C2 * 4 * 4, configs.num_class)
        elif self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(self.C2, self.pred_len * configs.c_out)
        else:
            raise NotImplementedError(
                f"RP-CNN 為關係影像分類模型，未支援 task_name='{self.task_name}'"
                "（建議用於 classification，亦可用於 forecast）。"
            )

    def _to_image(self, x_enc, eps=1e-8):
        # x_enc: [B, L, C] -> RP 影像 [B, C, L, L]
        x = x_enc.permute(0, 2, 1)                       # [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + eps)                     # 視窗 z-score 標準化
        diff = x.unsqueeze(-1) - x.unsqueeze(-2)         # [B, C, L, L]
        dist = diff.abs()                                # ||x_i - x_j|| (單變量)
        img = torch.exp(-self.gamma * dist)              # R_{i,j}
        return img

    def encoder(self, x_enc):
        img = self._to_image(x_enc)                      # [B, C, L, L]
        return self.features(img)                        # [B, C2, h, w]

    def classification(self, x_enc):
        feat = self.gap(self.encoder(x_enc))             # [B, C2, 4, 4]
        feat = feat.reshape(feat.size(0), -1)
        feat = self.dropout(feat)
        return self.projection(feat)                     # [B, num_class]

    def forecast(self, x_enc):
        # RevIN 實例正規化（可學習仿射），影像化前做 'norm'
        x_enc = self.revin_layer(x_enc, 'norm')

        feat = self.gap(self.encoder(x_enc))             # [B, C2, 1, 1]
        feat = feat.reshape(feat.size(0), -1)            # [B, C2]
        out = self.projection(feat)                      # [B, pred_len * c_out]
        out = out.reshape(out.size(0), self.pred_len, self.c_out)

        # RevIN 反正規化（自動對齊輸出通道數 c_out）
        out = self.revin_layer(out, 'denorm')
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc)            # [B, num_class]
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc)[:, -self.pred_len:, :]  # [B, L, D]
        return None
