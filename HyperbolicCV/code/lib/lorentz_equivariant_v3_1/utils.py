import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_group_channels(x, num_group):
    # print("x", x.shape[-1] )
    # print(x)
    x_space = x.narrow(-1, 1, x.shape[-1] - 1).to(device)
    x_time = x.narrow(-1, 0, 1).to(device)
    indices = torch.arange(0, x_space.size()[-1], step=num_group).to(device)
    x_space_original = torch.index_select(x_space, dim=-1, index=indices).to(device)
    x_original = torch.cat([x_time, x_space_original], dim=-1)

    return x_space, x_time, x_space_original, x_original