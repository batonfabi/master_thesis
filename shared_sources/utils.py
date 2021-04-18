import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake, device="cpu"):
    # calculates (||∇_x̂ D_w(x̂)||_2 − 1)^2
    
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

    # equals x̂
    interpolated_images = epsilon * real + fake * (1 - epsilon)

    # equals D_w(x̂)
    mixed_scores = critic(interpolated_images)

    # calculate gradient ∇_x̂ D_w(x̂)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    
    # ||∇_x̂ D_w(x̂)||_2
    gradient_norm = gradient.norm(2, dim=1)
    
    # (||∇_x̂ D_w(x̂)||_2 − 1)^2
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty