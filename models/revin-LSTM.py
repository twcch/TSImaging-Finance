import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    LSTM + RevIN（對應 LSTM 基準的 RevIN 版本）。

    結構與 models/LSTM.py 完全相同，唯一差異在於預測（forecast）路徑的實例
    正規化：以可學習仿射的 RevIN（Reversible Instance Normalization）取代原本
    固定式的非平穩正規化（NSN）。RevIN 在輸入端做 'norm'、輸出端做 'denorm'，
    並保留每通道可學習的 scale/shift，較能適應金融序列的非平穩性。

    與 LSTM 共用相同的呼叫介面：
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    分類 / 重建類任務維持與 LSTM 基準一致（不額外加入 RevIN）。

    註：RevIN 自行處理正規化，故與 run.py 的 --use_norm 旗標無關。

    Config 對應：
        enc_in   -> LSTM 輸入特徵維度（亦為 RevIN 通道數）
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

        # RevIN 實例正規化（可學習仿射），取代固定式 NSN
        self.revin_layer = RevIN(configs.enc_in)

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
        # RevIN 實例正規化（可學習仿射），取代固定式 NSN
        x_enc = self.revin_layer(x_enc, 'norm')

        lstm_out = self.encoder(x_enc)
        last_step = lstm_out[:, -1, :]                      # [B, d_model]
        dec_out = self.projection(last_step)               # [B, pred_len * c_out]
        dec_out = dec_out.reshape(dec_out.size(0), self.pred_len, -1)

        # RevIN 反正規化（自動對齊輸出通道數 c_out）
        dec_out = self.revin_layer(dec_out, 'denorm')
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
        # 以 padding mask 將補零時間步歸零（與 LSTM 分類流程一致）
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
