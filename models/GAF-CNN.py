import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    GAF-CNN：角度關係影像化模型（對應論文 Table 1 的 "GAF-CNN"）。

    將一維報酬率視窗以 Gramian Angular Summation Field (GASF) 轉為二維影像，
    再以簡單 2D-CNN 分類。影像化在 forward 內即時計算，故輸入仍與其他模型相
    同（[B, seq_len, enc_in]），可直接套用本專案既有的分類資料流程。

    GASF 轉換（論文式 4–6）：
        x̃_i = 2(x_i - min(X)) / (max(X) - min(X)) - 1
        φ_i = arccos(x̃_i)
        G_{i,j} = cos(φ_i + φ_j)
    每個輸入變數產生一張 L×L 影像，堆疊為 [B, enc_in, L, L] 後輸入 CNN。

    CNN 架構（論文 §3.4，簡單影像模型）：
        兩層 Conv-BN-ReLU-MaxPool -> Dropout -> 全連接輸出
    輸出為 num_class 個 logits，搭配框架的 CrossEntropyLoss。

    Config 對應：
        enc_in   -> 影像通道數（輸入變數數）
        dropout  -> 分類頭 dropout
        c_out    -> 預測任務輸出變數數
        num_class-> 分類類別數（由 Exp_Classification 動態注入）
    """

    C1, C2 = 16, 32  # 兩層卷積的通道數（簡單影像模型）

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.seq_len if self.task_name == 'classification' else configs.pred_len
        self.c_out = configs.c_out

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
                f"GAF-CNN 為關係影像分類模型，未支援 task_name='{self.task_name}'"
                "（建議用於 classification，亦可用於 forecast）。"
            )

    @staticmethod
    def _to_image(x_enc, eps=1e-8):
        # x_enc: [B, L, C] -> GASF 影像 [B, C, L, L]
        x = x_enc.permute(0, 2, 1)                       # [B, C, L]
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
        x_tilde = 2 * (x - x_min) / (x_max - x_min + eps) - 1
        x_tilde = x_tilde.clamp(-1 + 1e-6, 1 - 1e-6)     # arccos 數值安全
        phi = torch.arccos(x_tilde)                      # [B, C, L]
        # G_{i,j} = cos(phi_i + phi_j)
        img = torch.cos(phi.unsqueeze(-1) + phi.unsqueeze(-2))  # [B, C, L, L]
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
        feat = self.gap(self.encoder(x_enc))             # [B, C2, 1, 1]
        feat = feat.reshape(feat.size(0), -1)            # [B, C2]
        out = self.projection(feat)                      # [B, pred_len * c_out]
        return out.reshape(out.size(0), self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc)            # [B, num_class]
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc)[:, -self.pred_len:, :]  # [B, L, D]
        return None
