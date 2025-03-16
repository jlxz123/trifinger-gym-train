import torch

if torch.cuda.is_available():
    print("everything is okay")
else:
    print("maybe your cuda is not available")