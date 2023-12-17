import torch
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
import pandas as pd


def get_splines(X, device):
    X = X.permute(1, 0)
    indices = torch.linspace(0, 1, X.shape[0]).to(device)
    splines = []
    for i in range(X.shape[1]):
        the_X = X[:,i:i+1]
        coeffs = natural_cubic_spline_coeffs(indices, the_X)
        spline = NaturalCubicSpline(coeffs)
        splines.append(spline)
    return splines


if __name__ == '__main__':
    data = pd.read_csv("data/dataset_525_871.csv").iloc[0:3,0:-1].to_numpy()
    data = torch.tensor(data)
    splines = get_splines(data, "cpu")
    idx1 = torch.tensor([0.1,0.9])
    idx2 = torch.tensor([0.2,0.8])
    idx3 = torch.tensor([0.3,0.7])
    print(splines[0].evaluate(idx1))
    print(splines[1].evaluate(idx2))
    print(splines[2].evaluate(idx3))

    # tensor([[0.6815],
    #         [1.2184]], dtype=torch.float64)
    # tensor([[0.9418],
    #         [1.2749]], dtype=torch.float64)
    # tensor([[1.2890],
    #         [1.6217]], dtype=torch.float64)