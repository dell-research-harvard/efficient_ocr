import torch
import timm
from transformers import AutoModel


def AutoEncoderFactory(backend, modelpath):

    if backend == "timm":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = timm.create_model(model, num_classes=0, pretrained=True)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                return x 

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    elif backend == "hf":

        class AutoEncoder(torch.nn.Module):

            def __init__(self, model=modelpath, device='cuda'):
                super().__init__()
                net = AutoModel.from_pretrained(model)
                net.to(device)
                self.net = net

            def forward(self, x):
                x = self.net(x)
                x = x.last_hidden_state[:,0,:]
                return x

            @classmethod
            def load(cls, checkpoint):
                ptnet = cls()
                ptnet.load_state_dict(torch.load(checkpoint))
                return ptnet

    else:
        
        raise NotImplementedError

    return AutoEncoder