import torch

from typing import Optional, Tuple
import numpy as np

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, PointLights,
    RasterizationSettings,
    MeshRasterizer, MeshRenderer,
    SoftPhongShader, SoftSilhouetteShader
)


class ImageRenderer(object):
    """
    Renders image from mesh.
    """

    def __init__(self,
                 image_size: int,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        self.image_size = image_size

        # camera
        self._distance = 2.7
        self._elevation = 10
        self._azimuth = 150
        self._cameras = FoVPerspectiveCameras(device=self.device)

        # lights
        self._lights: Optional[PointLights] = None
        self._light_location = [[0.0, 0.0, -3.0]]
        self._initialise_lights()

        # renderer
        self.r_textured: MeshRenderer = None
        self._initialise_renderer()
        self.camera_params = self.camera_params

    def _initialise_lights(self) -> None:
        self._lights = PointLights(device=self.device,
                                   location=self.light_location)

    def _initialise_renderer(self) -> None:
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

    def render(self, mesh, camera_params: Optional[list] = None) -> np.ndarray:
        if camera_params:
            self.camera_params = camera_params
        return self._get_rendered_image(mesh)

    def _get_rendered_image(self, mesh) -> np.ndarray:
        image_tensor = self.r_textured(mesh)
        np_image = image_tensor[0, ..., :3].cpu().numpy() * 255

        return np.uint8(np_image)

    @property
    def light_location(self) -> list:
        return self._light_location

    @light_location.setter
    def light_location(self, value: list) -> None:
        """ Setting the light location automatically updates the lights object."""
        self._light_location = value
        self.lights = PointLights(device=self.device,
                                  location=self.light_location)

    @property
    def lights(self) -> PointLights:
        return self._lights

    @lights.setter
    def lights(self, value: PointLights) -> None:
        """ Setting the lights automatically updates the renderer object. """
        self._lights = value
        if self.r_textured:
            self.r_textured.shader.lights = self._lights

    @property
    def camera_params(self) -> Tuple[float, float, float]:
        return self._distance, self._elevation, self._azimuth

    @camera_params.setter
    def camera_params(self, value: list) -> None:
        """ Setting the camera params automatically updates the camera object. """
        self._distance, self._elevation, self._azimuth = value

        rot, trans = look_at_view_transform(*self.camera_params)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=rot, T=trans)

    @property
    def cameras(self) -> FoVPerspectiveCameras:
        return self._cameras

    @cameras.setter
    def cameras(self, value: FoVPerspectiveCameras) -> None:
        """ Setting the camera value automatically updates the renderer object. """
        self._cameras = value
        if self.r_textured:
            self.r_textured.rasterizer.cameras = value
            self.r_textured.shader.cameras = value


class ImageRendererDynamic(ImageRenderer):
    """
    Renders image from mesh.
    Lights automatically follow the camera.
    """
    def __init__(self, *args, **kwargs):
        super(ImageRendererDynamic, self).__init__(*args, **kwargs)

    @property
    def light_location(self):
        """ The light location is the camera centre. """
        return self._cameras.get_camera_center()

    @light_location.setter
    def light_location(self, value: list) -> None:
        ImageRenderer.light_location.fset(self, value)

    @ImageRenderer.camera_params.setter
    def camera_params(self, value: list) -> None:
        """ Updating the camera parameters involves relocating the light. """
        ImageRenderer.camera_params.fset(self, value)
        self.light_location = self.light_location


class ImageRendererStatic(ImageRenderer):
    """
    Renders image from mesh.
    Has a silhouette renderer in addition to the textured renderer.
    """
    def __init__(self, *args, **kwargs):
        self.r_textured: Optional[MeshRenderer] = None
        self.r_silhouette: Optional[MeshRenderer] = None

        super(ImageRendererStatic, self).__init__(*args, **kwargs)

        self._initialise_renderer()

    def __call__(self, *args,
                 silhouette: bool = False, **kwargs):
        kwargs.setdefault('silhouette', silhouette)
        if silhouette:
            return self.r_silhouette(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def _initialise_renderer(self) -> None:
        sigma = 1e-4
        raster_silhouette = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )

        # differentiable soft renderer using per vertex RGB for texture
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
