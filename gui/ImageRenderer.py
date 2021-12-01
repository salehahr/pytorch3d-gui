import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    RasterizationSettings, PointLights, MeshRasterizer, MeshRenderer, SoftPhongShader
from pytorch3d.utils import ico_sphere


class ImageRenderer(object):
    """
    Renders image from mesh.
    Lights automatically follow the camera.
    """

    def __init__(self,
                 image_size: int,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        self.image_size = image_size
        self._is_loaded = False

        # camera
        self._distance = 2.7
        self._elevation = 10
        self._azimuth = 150
        self._cameras = FoVPerspectiveCameras(device=self.device)

        # lights
        self._lights = None
        self._initialise_lights()

        # renderer
        self.diff_renderer = None
        self._raster_textured = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.renderer = None
        self._initialise_renderer()

        # mesh
        self._mesh = None

    def _initialise_lights(self):
        self._lights = PointLights(device=self.device,
                                   location=self.light_location)

    def _initialise_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._cameras,
                raster_settings=self._raster_textured
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self._cameras,
                lights=self._lights
            )
        )
        self.camera_params = self.camera_params

    def __call__(self, *args, **kwargs):
        # TODO: subclass MeshRenderer
        return self.renderer(*args, **kwargs)

    def load_file(self, filepath: str):
        self.mesh = load_objs_as_meshes([filepath], device=self.device)

    def load_sphere(self):
        self.mesh = ico_sphere(4, self.device)

    def render(self, camera_params=None):
        if camera_params:
            self.camera_params = camera_params
        return self._get_rendered_image()

    def _get_rendered_image(self):
        image_tensor = self.renderer(self.mesh)
        np_image = image_tensor[0, ..., :3].cpu().numpy() * 255

        return np.uint8(np_image)

    def get_texture_map(self) -> np.ndarray:
        texture_image = self.mesh.textures.maps_padded() * 255
        np_img = texture_image.squeeze().cpu().numpy().astype(np.uint8)

        self.image_size = max(np_img.shape[0:2])

        return np_img

    @property
    def light_location(self):
        return self._cameras.get_camera_center()

    def _set_light_location(self):
        self.lights = PointLights(device=self.device,
                                  location=self.light_location)

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        self._lights = value
        if self.renderer:
            self.renderer.shader.lights = self._lights

    @property
    def camera_params(self):
        return self._distance, self._elevation, self._azimuth

    @camera_params.setter
    def camera_params(self, value: list):
        self._distance, self._elevation, self._azimuth = value

        rot, trans = look_at_view_transform(*self.camera_params)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=rot, T=trans)
        self._set_light_location()

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value
        if self.renderer:
            self.renderer.rasterizer.cameras = value
            self.renderer.shader.cameras = value

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self.is_loaded = True

    @property
    def is_loaded(self):
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self, value):
        self._is_loaded = value

    @property
    def num_views_for_diff_render(self):
        return self.diff_renderer.num_views
