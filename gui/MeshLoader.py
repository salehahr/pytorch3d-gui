import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    RasterizationSettings, PointLights, MeshRasterizer, MeshRenderer, SoftPhongShader


class MeshLoader(object):
    def __init__(self,
                 device: str = 'cuda:0'):
        self._device = torch.device(device)
        torch.cuda.set_device(self._device)

        self._image_size: int = 512
        self._is_loaded = False

        # camera
        self._distance = 2.7
        self._elevation = 10
        self._azimuth = 150
        self._cameras = FoVPerspectiveCameras(device=self._device)

        # _lights
        self._lights = None
        self._initialise_lights()

        # renderer
        self._raster_settings = RasterizationSettings(
            image_size=self._image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self._renderer = None
        self._initialise_renderer()

        # mesh
        self._mesh = None

    def _initialise_lights(self):
        self._lights = PointLights(device=self._device,
                                   location=self.light_location)

    def _initialise_renderer(self):
        self._renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._cameras,
                raster_settings=self._raster_settings
            ),
            shader=SoftPhongShader(
                device=self._device,
                cameras=self._cameras,
                lights=self._lights
            )
        )
        self.camera_params = self.camera_params

    def load_file(self, filepath: str):
        self._mesh = load_objs_as_meshes([filepath], device=self._device)
        self._is_loaded = True

    def render(self, camera_params=None):
        if camera_params:
            self.camera_params = camera_params
        return self._get_rendered_image()

    def _get_rendered_image(self):
        image_tensor = self._renderer(self._mesh)
        np_image = image_tensor[0, ..., :3].cpu().numpy() * 255

        return np.uint8(np_image)

    def get_texture_map(self) -> np.ndarray:
        texture_image = self._mesh.textures.maps_padded() * 255
        np_img = texture_image.squeeze().cpu().numpy().astype(np.uint8)

        self._image_size = max(np_img.shape[0:2])

        return np_img

    @property
    def light_location(self):
        return self._cameras.get_camera_center()

    def _set_light_location(self):
        self.lights = PointLights(device=self._device,
                                  location=self.light_location)

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        self._lights = value
        if self._renderer:
            self._renderer.shader.lights = self._lights

    @property
    def camera_params(self):
        return self._distance, self._elevation, self._azimuth

    @camera_params.setter
    def camera_params(self, value: list):
        self._distance, self._elevation, self._azimuth = value

        rot, trans = look_at_view_transform(*self.camera_params)
        self.cameras = FoVPerspectiveCameras(device=self._device, R=rot, T=trans)
        self._set_light_location()

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value
        if self._renderer:
            self._renderer.rasterizer.cameras = value
            self._renderer.shader.cameras = value

    @property
    def is_loaded(self):
        return self._is_loaded