
from torch import nn




"""GRU"""
class GRU(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(GRU, self).__init__()
        """:param"""
        self.dropout=0.5


        self.gru = nn.GRU(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,)

        self.out = nn.Linear(hidden_size, pre_len)

    def forward(self, x):
        temp, _ = self.gru(x)
        s, b, h = temp.size()
        temp = temp.reshape(s * b, h)
        outs = self.out(temp)
        gru_out = outs.reshape(s, b, -1)
        return gru_out

#
"""BiLSTM"""
class LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(LSTM, self).__init__()
        """:param"""
        self.dropout=0.5


        self.lstm = nn.LSTM(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True)
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):
        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.reshape(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.reshape(s, b, -1)
        return lstm_out