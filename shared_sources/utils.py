""" 
inspired by  https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/train.py
MIT License

Copyright (c) 2020 Aladdin Persson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. """

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