import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer, device="cpu"):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        score = out.squeeze()
        if score.ndim == 0:
            score = score.unsqueeze(0)
        score.backward(retain_graph=True)

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)  # [B, H, W]
        cam = F.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
