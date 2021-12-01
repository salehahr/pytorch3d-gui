import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    RasterizationSettings, PointLights, MeshRasterizer, MeshRenderer, SoftPhongShader, SoftSilhouetteShader
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

        # mesh
        self._mesh = None

        # camera
        self._distance = 2.7
        self._elevation = 10
        self._azimuth = 150
        self._cameras = FoVPerspectiveCameras(device=self.device)

        # lights
        self._lights = None
        self._light_location = [[0.0, 0.0, -3.0]]
        self._initialise_lights()

        # renderer
        self.r_textured = None
        self._initialise_renderer()
        self.camera_params = self.camera_params

    def _initialise_lights(self):
        self._lights = PointLights(device=self.device,
                                   location=self.light_location)

    def _initialise_renderer(self):
        raster_textured = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.r_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._cameras,
                raster_settings=raster_textured
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self._cameras,
                lights=self._lights
            )
        )

    def __call__(self, *args, **kwargs):
        # TODO: subclass MeshRenderer
        return self.r_textured(*args, **kwargs)

    def load_file(self, filepath: str):
        self.mesh = load_objs_as_meshes([filepath], device=self.device)

    def load_sphere(self):
        self.mesh = ico_sphere(4, self.device)

    def render(self, camera_params=None):
        if camera_params:
            self.camera_params = camera_params
        return self._get_rendered_image()

    def _get_rendered_image(self):
        image_tensor = self.r_textured(self.mesh)
        np_image = image_tensor[0, ..., :3].cpu().numpy() * 255

        return np.uint8(np_image)

    def get_texture_map(self) -> np.ndarray:
        texture_image = self.mesh.textures.maps_padded() * 255
        np_img = texture_image.squeeze().cpu().numpy().astype(np.uint8)

        self.image_size = max(np_img.shape[0:2])

        return np_img

    @property
    def light_location(self):
        return self._light_location

    @light_location.setter
    def light_location(self, value):
        self._light_location = value

    @property
    def lights(self):
        return self._lights

    @lights.setter
    def lights(self, value):
        self._lights = value
        if self.r_textured:
            self.r_textured.shader.lights = self._lights

    @property
    def camera_params(self):
        return self._distance, self._elevation, self._azimuth

    @camera_params.setter
    def camera_params(self, value: list):
        self._distance, self._elevation, self._azimuth = value

        rot, trans = look_at_view_transform(*self.camera_params)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=rot, T=trans)

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value
        if self.r_textured:
            self.r_textured.rasterizer.cameras = value
            self.r_textured.shader.cameras = value

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self._is_loaded = True if value else False

    @property
    def is_loaded(self):
        return self._is_loaded


class ImageRendererDynamic(ImageRenderer):
    def __init__(self, filepath, *args, **kwargs):
        super(ImageRendererDynamic, self).__init__(*args, **kwargs)
        self.load_file(filepath)

    @property
    def light_location(self):
        return self._cameras.get_camera_center()

    @light_location.setter
    def light_location(self, value):
        ImageRenderer.light_location.fset(self, value)
        self.lights = PointLights(device=self.device,
                                  location=self.light_location)

    @ImageRenderer.camera_params.setter
    def camera_params(self, value):
        ImageRenderer.camera_params.fset(self, value)
        self.light_location = self.light_location


class ImageRendererStatic(ImageRenderer):
    def __init__(self, *args, **kwargs):
        self.r_textured = None
        self.r_silhouette = None

        super(ImageRendererStatic, self).__init__(*args, **kwargs)

        self._initialise_renderer()

    def __call__(self, *args,
                 silhouette: bool = False, **kwargs):
        kwargs.setdefault('silhouette', silhouette)
        if silhouette:
            return self.r_silhouette(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def _initialise_renderer(self):
        sigma = 1e-4
        raster_silhouette = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )

        renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_silhouette),
            shader=SoftPhongShader(device=self.device,
                                   lights=self.lights)
        )
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_silhouette),
            shader=SoftSilhouetteShader()
        )

        self.r_textured = renderer_textured
        self.r_silhouette = renderer_silhouette
