from torch import nn


def get_fc(in_features, widths, classes):
    layers = []
    for out_features in widths:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, classes))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model


