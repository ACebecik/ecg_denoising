"""
This file implements the classification model CNN.
"""
import torch
import torch.nn as nn


class CnnClassifier(nn.Module):
    def __init__(self, in_channels=1, kernel_sizes=None, num_filters=None,
                 p_dropout= 0.3, fc_size=1024, window_size = 1024):

        super(CnnClassifier, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [7,7,7,7]
        if num_filters is None:
            num_filters = [32,32,64,64]

        assert len(kernel_sizes) == len(num_filters), "Provided kernel sizes and num filters not compatible!"

        self.window_size = window_size

        layers = []
        in_ch = in_channels
        for size,num in zip(kernel_sizes, num_filters):
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=num, kernel_size=size, padding=size //2 ))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(p=p_dropout))
            in_ch = num

        self.conv_section = nn.Sequential(*layers)

        # infer the output size for feeding the fc
        with torch.no_grad():
            dummy_input = torch.zeros((1,in_channels, window_size))
            dummy_processed = self.conv_section(dummy_input)
            flattened_size = dummy_processed.reshape(1,-1).shape[1]

        self.fc_section = nn.Sequential(nn.Linear(in_features=flattened_size, out_features=fc_size),
                                        nn.ReLU(),
                                        nn.Linear(in_features=fc_size, out_features=1))


    def forward(self, x):
        assert x.shape[2] == self.window_size, f"Expected input length {self.window_size}, got {x.shape[2]}"""
        x = self.conv_section(x)
        x = x.reshape(x.shape[0],-1)
        return self.fc_section(x)

if __name__ == "__main__":
    model = CnnClassifier()
    dummy_inp = torch.ones((1024, 1, 1024))  # batch of 8
    output = model(dummy_inp)
    print("Output shape:", output.shape)  # should be (8, 1)



