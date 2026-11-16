"""Training-only modules. Frozen photo classifier + learned landmark branch."""
import torch
from torch import nn

class LandmarkFusionHead(nn.Module):
    def __init__(self,image_weight,image_bias,geometry_mean,geometry_std):
        super().__init__()
        self.register_buffer('image_weight',image_weight.detach().clone())
        self.register_buffer('image_bias',image_bias.detach().clone())
        self.register_buffer('geometry_mean',geometry_mean.detach().clone())
        self.register_buffer('geometry_std',geometry_std.detach().clone().clamp_min(.05))
        self.geometry_encoder=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.geometry_weight=nn.Parameter(torch.zeros(8,16))
    def geometry_delta(self,landmarks):
        encoded=self.geometry_encoder((landmarks-self.geometry_mean)/self.geometry_std)
        return torch.nn.functional.linear(encoded,self.geometry_weight)
    def forward(self,features,landmarks):
        encoded=self.geometry_encoder((landmarks-self.geometry_mean)/self.geometry_std)
        fused=torch.cat([features,encoded],dim=-1)
        weight=torch.cat([self.image_weight,self.geometry_weight],dim=1)
        return torch.nn.functional.linear(fused,weight,self.image_bias)

class PhotoAndLandmarks(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone;self.head=head
        for p in self.backbone.parameters():p.requires_grad=False
        self.backbone.eval()
    def forward(self,image,landmarks):
        features=self.backbone.forward_head(self.backbone.forward_features(image),pre_logits=True)
        return self.head(features,landmarks)
