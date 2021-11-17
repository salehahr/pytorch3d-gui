import torch
import numpy as np

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    RasterizationSettings, PointLights, MeshRasterizer, MeshRenderer, SoftPhongShader


class MeshLoader(object):
    def __init__(self,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        self.image_size: int = 0

        # camera
        R, T = look_at_view_transform(dist=2.7, elev=0, azim=180)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # lights
        self.lights = PointLights(device=self.device,
                                  location=[[0.0, 0.0, -3.0]])

    def get_texture_map(self, filepath: str) -> np.ndarray:
        mesh = self._get_mesh(filepath)
        texture_image = mesh.textures.maps_padded() * 255
        np_img = texture_image.squeeze().cpu().numpy().astype(np.uint8)

        self.image_size = max(np_img.shape[0:2])

        return np_img

    def get_rendered_image(self, filepath: str):
        mesh = self._get_mesh(filepath)

        image_size = 512
        renderer = self._generate_renderer(image_size)

        image_tensor = renderer(mesh)
        np_image = image_tensor[0, ..., :3].cpu().numpy() * 255

        return np.uint8(np_image)

    def _get_mesh(self, filepath: str):
        return load_objs_as_meshes([filepath], device=self.device)

    def _generate_renderer(self, image_size: int):
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights
            )
        )
