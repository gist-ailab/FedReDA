import torch

var_clamp = 1e-5


def compute_var(mean, sq_mean):
    var_dict = {}
    for k in mean.keys():
        var = torch.clamp(sq_mean[k] - mean[k] ** 2, var_clamp)
        var_dict[k] = var
    return var_dict


# def compute_mean_sq(teachers, base_model):
#     w_avg = {}
#     w_sq_avg = {}
#     w_norm = {}
#     for k in teachers[0].keys():
#         if "batches_tracked" in k: continue
#         w_avg[k] = torch.zeros(teachers[0][k].size())
#         w_sq_avg[k] = torch.zeros(teachers[0][k].size())
#         w_norm[k] = 0.0
#     for k in w_avg.keys():
#         if "batches_tracked" in k: continue
#         for i in range(0, len(teachers)):
#             grad = teachers[i][k].cpu() - base_model[k].cpu()
#             norm = torch.norm(grad, p=2)
#             grad = grad / norm
#             sq_grad = grad ** 2
#             w_avg[k] += grad
#             w_sq_avg[k] += sq_grad
#             w_norm[k] += norm
#         w_avg[k] = torch.div(w_avg[k], len(teachers))
#         w_sq_avg[k] = torch.div(w_sq_avg[k], len(teachers))
#         w_norm[k] = torch.div(w_norm[k], len(teachers))
#     return w_avg, w_sq_avg, w_norm

def compute_mean_sq(teachers, base_model):
    device = next(iter(base_model.values())).device  # base_model의 디바이스

    w_avg = {}
    w_sq_avg = {}
    w_norm = {}
    for k in teachers[0].keys():
        if not (("reins" in k) or ("linear" in k)):
            continue
        w_avg[k] = torch.zeros_like(teachers[0][k]).to(device)
        w_sq_avg[k] = torch.zeros_like(teachers[0][k]).to(device)
        w_norm[k] = 0.0

    for k in w_avg.keys():
        for i in range(len(teachers)):
            t_weight = teachers[i][k].to(device)
            b_weight = base_model[k].to(device)

            grad = t_weight - b_weight
            norm = torch.norm(grad, p=2)
            if norm < 1e-12 or torch.isnan(norm):
                continue
            grad = grad / norm
            sq_grad = grad ** 2
            w_avg[k] += grad
            w_sq_avg[k] += sq_grad
            w_norm[k] += norm
        w_avg[k] /= len(teachers)
        w_sq_avg[k] /= len(teachers)
        w_norm[k] /= len(teachers)
    return w_avg, w_sq_avg, w_norm