import random
import torch
import math

def distribute_padding(x, padding_idx=1, after_idx=0):
    '''Experimentation with distributed padding, not used in the final approach.'''

    padding_count = (x[:, after_idx:] == padding_idx).sum(dim=-1)
    device = x.device
    x_list = x.unsqueeze(-1).tolist()

    batch_size = x.size(0)
    # max_len = x.size(1)
    for i in range(batch_size):
        x_list[i] = [
            t for j, t in enumerate(x_list[i]) if t[0] != padding_idx or j < after_idx
        ]

    for i in range(batch_size):
        for p in range(padding_count[i]):
            chosen = random.randint(after_idx, len(x_list[i]) - 1)
            x_list[i][chosen].append(padding_idx)

        x_list[i] = [torch.tensor(t) for t in x_list[i]]
        x_list[i] = torch.concat(x_list[i], dim=-1)
    x_list = torch.stack(x_list, dim=0).to(device)

    return x_list


def get_time_variables_old_schedule(
    t, total_t, device
):  # according to https://arxiv.org/pdf/2102.09672.pdf
    '''Old cosine Schedule'''

    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t - 1, total_t) / ft(
        torch.zeros(t.shape).to(device), total_t
    )
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


def get_time_variables_new_schedule(t, total_t, device):
    '''ParaGuide Schedule'''

    def ft(small_t, big_t, s=1e-4):
        return torch.sqrt((big_t - small_t) / big_t)

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t - 1, total_t) / ft(
        torch.zeros(t.shape).to(device), total_t
    )
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t
