import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    LSTM baseline for the Time-Series-Library framework.

    一維序列深度學習基準（對應論文 Table 1 的 "LSTM"）。直接以一維報酬率
    序列為輸入，透過多層 LSTM 編碼時間依賴，再依任務別接上不同的輸出頭。

    與 TimesNet 共用相同的呼叫介面：
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    並依 configs.task_name 分派至對應任務。論文的「下一交易日漲跌方向預測」
    對應 task_name == 'classification'（二元分類）。

    Config 對應：
        enc_in   -> LSTM 輸入特徵維度
        d_model  -> LSTM 隱藏維度
        e_layers -> LSTM 堆疊層數
        dropout  -> 層間 / 分類頭 dropout
        c_out    -> 預測任務輸出變數數
        num_class-> 分類任務類別數（由 Exp_Classification 動態注入）
    """

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

        # LSTM 編碼器（單層時關閉 dropout 以避免 PyTorch warning）
        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
            batch_first=True,
        )

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            # 取最後隱藏狀態映射為整段預測視窗
            self.projection = nn.Linear(
                configs.d_model, self.pred_len * configs.c_out)
        if self.task_name in ('imputation', 'anomaly_detection'):
            # 逐步重建
            self.projection = nn.Linear(configs.d_model, configs.c_out)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        lstm_out, _ = self.lstm(x)
        return lstm_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 非平穩實例正規化（與 TimesNet 一致），緩解金融序列的非平穩性
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        lstm_out = self.encoder(x_enc)
        last_step = lstm_out[:, -1, :]                      # [B, d_model]
        dec_out = self.projection(last_step)               # [B, pred_len * c_out]
        dec_out = dec_out.reshape(dec_out.size(0), self.pred_len, -1)

        # 反正規化
        c_out = dec_out.size(-1)
        dec_out = dec_out * stdev[:, 0, :c_out].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :c_out].unsqueeze(1)
        return dec_out

    def imputation(self, x_enc):
        lstm_out = self.encoder(x_enc)
        return self.projection(lstm_out)                   # [B, seq_len, c_out]

    def anomaly_detection(self, x_enc):
        lstm_out = self.encoder(x_enc)
        return self.projection(lstm_out)                   # [B, seq_len, c_out]

    def classification(self, x_enc, x_mark_enc):
        lstm_out = self.encoder(x_enc)                     # [B, seq_len, d_model]
        output = self.act(lstm_out)
        output = self.dropout(output)
        # 以 padding mask 將補零時間步歸零（與 TimesNet 分類流程一致）
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
