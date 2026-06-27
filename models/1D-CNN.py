import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    1D-CNN baseline for the Time-Series-Library framework.

    一維局部型態基準（對應論文 Table 1 的 "1D-CNN"）。直接以一維報酬率序列
    為輸入，透過堆疊的一維卷積沿時間軸萃取局部型態（local pattern），再依任
    務別接上不同的輸出頭。卷積採 same padding（stride=1）以保留序列長度，
    故時間維與 padding mask 對齊。

    與 LSTM / TimesNet 共用相同的呼叫介面：
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    並依 configs.task_name 分派至對應任務。論文的「下一交易日漲跌方向預測」
    對應 task_name == 'classification'（二元分類）。

    Config 對應：
        enc_in   -> 卷積輸入通道數（輸入特徵維度）
        d_model  -> 卷積特徵通道數
        e_layers -> 卷積區塊堆疊層數
        dropout  -> 卷積區塊 / 分類頭 dropout
        c_out    -> 預測任務輸出變數數
        num_class-> 分類任務類別數（由 Exp_Classification 動態注入）
    """

    KERNEL_SIZE = 3  # 固定 kernel，保留局部感受野；padding=1 維持序列長度

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers

        # 分類 / 重建類任務以序列長度為輸出長度；預測類任務用 pred_len
        if self.task_name in ('classification', 'anomaly_detection', 'imputation'):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # 一維卷積編碼器（FCN 風格：Conv1d -> BatchNorm -> GELU -> Dropout）
        pad = self.KERNEL_SIZE // 2
        blocks = []
        in_ch = configs.enc_in
        for _ in range(max(1, self.num_layers)):
            blocks.append(nn.Conv1d(in_ch, configs.d_model,
                                    kernel_size=self.KERNEL_SIZE, padding=pad))
            blocks.append(nn.BatchNorm1d(configs.d_model))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(configs.dropout))
            in_ch = configs.d_model
        self.conv = nn.Sequential(*blocks)

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            # 時間維全域平均池化後映射為整段預測視窗
            self.projection = nn.Linear(
                configs.d_model, self.pred_len * configs.c_out)
        if self.task_name in ('imputation', 'anomaly_detection'):
            # 逐步重建
            self.projection = nn.Linear(configs.d_model, configs.c_out)
        if self.task_name == 'classification':
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        x = x.permute(0, 2, 1)          # [B, enc_in, seq_len]
        feat = self.conv(x)             # [B, d_model, seq_len]
        feat = feat.permute(0, 2, 1)    # [B, seq_len, d_model]
        return feat

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 非平穩實例正規化（與 LSTM / TimesNet 一致），緩解金融序列的非平穩性
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        feat = self.encoder(x_enc)                 # [B, seq_len, d_model]
        pooled = feat.mean(dim=1)                  # [B, d_model] 全域平均池化
        dec_out = self.projection(pooled)          # [B, pred_len * c_out]
        dec_out = dec_out.reshape(dec_out.size(0), self.pred_len, -1)

        # 反正規化
        c_out = dec_out.size(-1)
        dec_out = dec_out * stdev[:, 0, :c_out].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :c_out].unsqueeze(1)
        return dec_out

    def imputation(self, x_enc):
        feat = self.encoder(x_enc)
        return self.projection(feat)               # [B, seq_len, c_out]

    def anomaly_detection(self, x_enc):
        feat = self.encoder(x_enc)
        return self.projection(feat)               # [B, seq_len, c_out]

    def classification(self, x_enc, x_mark_enc):
        feat = self.encoder(x_enc)                 # [B, seq_len, d_model]
        output = self.dropout(feat)
        # 以 padding mask 將補零時間步歸零（與 LSTM / TimesNet 分類流程一致）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # (B, seq_len * d_model)
        output = output.reshape(output.shape[0], -1)
        # (B, num_class)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]          # [B, L, D]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)                  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)           # [B, L, D]
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)  # [B, num_class]
        return None
