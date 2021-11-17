import torch
import numpy as np

from pytorch3d.io import load_obj, load_objs_as_meshes


class MeshLoader(object):
    def __init__(self,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

    def get_texture_map(self, filepath: str) -> np.ndarray:
        # mesh = load_obj(filepath, device=self.device)
        mesh2 = load_objs_as_meshes([filepath], device=self.device)
        texture_image = mesh2.textures.maps_padded() * 255
        return texture_image.squeeze().cpu().numpy().astype(np.uint8)
