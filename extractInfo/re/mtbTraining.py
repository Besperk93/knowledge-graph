import torch

if torch.cuda.is_available():
    print("RUNNING ON GPU")
    device = torch.device("cuda")
else:
    print("NO GPU AVAILABLE")
    device = torch.device("cpu")
