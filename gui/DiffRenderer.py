import torch
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d.renderer import look_at_view_transform, \
    FoVPerspectiveCameras, PointLights, \
    RasterizationSettings, \
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex, SoftSilhouetteShader
from pytorch3d.utils import ico_sphere


def scale_and_normalise(mesh):
    verts = mesh.verts_packed()

    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])

    new_mesh = mesh.offset_verts(-center)
    new_mesh.scale_verts_((1.0 / float(scale)))

    return new_mesh


class DiffRenderer(object):
    def __init__(self,
                 mesh_loader,
                 device: str = 'cuda:0'):
        self.device = torch.device(device)

        # initial mesh
        initial_mesh = ico_sphere(4, device)
        initial_vertices_dim = initial_mesh.verts_packed().shape
        num_vertices = initial_vertices_dim[0]
        self._initial_mesh = initial_mesh

        # target mesh
        self._mesh_loader = mesh_loader
        self._target_mesh = None

        self.num_views = 20

        # renderer
        self._default_camera = None
        self.target_cameras = None
        self.lights = PointLights(device=self.device,
                                  location=[[0.0, 0.0, -3.0]])
        self.renderer_textured = None
        self.renderer_silhouette = None
        self._init_renderers()

        # initial values
        self.deform_verts = torch.full(initial_vertices_dim,
                                       0.0,
                                       device=device,
                                       requires_grad=True)
        self.sphere_verts_rgb = torch.full([1, num_vertices, 3], 0.5,
                                           device=device, requires_grad=True)

        # optimiser
        self.num_views_per_iteration = 2
        self.num_iter = 100
        self.plot_period = 50
        self.optimiser = torch.optim.SGD([self.deform_verts,
                                          self.sphere_verts_rgb],
                                         lr=1.0, momentum=0.9)

    def _init_renderers(self):
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )

        renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftPhongShader(device=self.device,
                                   lights=self.lights)
        )
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_silhouette),
            shader=SoftSilhouetteShader()
        )

        self.renderer_textured = renderer_textured
        self.renderer_silhouette = renderer_silhouette

        self.losses = {"rgb": {"weight": 1.0, "values": []},
                       "silhouette": {"weight": 1.0, "values": []},
                       "edge": {"weight": 1.0, "values": []},
                       "normal": {"weight": 0.01, "values": []},
                       "laplacian": {"weight": 1.0, "values": []},
                       }

    def render(self):
        target_cameras, target_silhouette, target_rgb = self.generate_views()
        self.optimise(target_cameras, target_silhouette, target_rgb)

    def generate_views(self):
        self.target_mesh = self._mesh_loader.mesh
        target_meshes = self._target_mesh.extend(self.num_views)

        elev = torch.linspace(0, 360, self.num_views)
        azim = torch.linspace(-180, 180, self.num_views)
        rots, trans = look_at_view_transform(dist=2.7, elev=elev, azim=azim)

        cameras = FoVPerspectiveCameras(device=self.device,
                                        R=rots, T=trans)
        self.default_camera = FoVPerspectiveCameras(device=self.device,
                                                    R=rots[None, 1, ...],
                                                    T=trans[None, 1, ...])
        target_cameras = [FoVPerspectiveCameras(device=self.device,
                                                     R=rots[None, i, ...],
                                                     T=trans[None, i, ...])
                               for i in range(self.num_views)]

        target_images = self.image_renderer(target_meshes,
                                            cameras=cameras,
                                            lights=self.lights)
        silhouette_images = self.renderer_silhouette(target_meshes,
                                                     cameras=cameras,
                                                     lights=self.lights)

        target_silhouette = [silhouette_images[i, ..., 3]
                             for i in range(self.num_views)]

        target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]

        return target_cameras, target_silhouette, target_rgb

    def optimise(self, target_cameras, target_silhouette, target_rgb):
        loop = tqdm(range(self.num_iter))
        for i in loop:
            self.optimiser.zero_grad()

            deformed_mesh = self._initial_mesh.offset_verts(self.deform_verts)
            deformed_mesh.textures = TexturesVertex(verts_features=self.sphere_verts_rgb)

            loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}
            loss["edge"] = mesh_edge_loss(deformed_mesh)
            loss["normal"] = mesh_normal_consistency(deformed_mesh)
            loss["laplacian"] = mesh_laplacian_smoothing(deformed_mesh, method="uniform")

            for j in self.random_views:
                predicted_images = self.renderer_textured(deformed_mesh,
                                                          cameras=target_cameras[j],
                                                          lights=self.lights)

                predicted_silhouette = predicted_images[..., 3]
                loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
                loss["silhouette"] += loss_silhouette / self.num_views_per_iteration

                predicted_rgb = predicted_images[..., :3]
                loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
                loss["rgb"] += loss_rgb / self.num_views_per_iteration

            sum_loss = torch.tensor(0.0, device=self.device)
            for k, l in loss.items():
                sum_loss += l * self.losses[k]["weight"]
                self.losses[k]["values"].append(float(l.detach().cpu()))

            loop.set_description("total_loss = %.6f" % sum_loss)

            if i % self.plot_period == 0:
                self.render_predicted_image(deformed_mesh, title="iter: %d" % i)
                plt.show()

            sum_loss.backward()
            self.optimiser.step()

    def render_predicted_image(self, predicted_mesh, title):
        inds = range(3)

        with torch.no_grad():
            predicted_images = self.renderer_textured(predicted_mesh)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
        plt.title(title)
        plt.axis("off")

    @property
    def mesh_loader(self):
        return self._mesh_loader

    @mesh_loader.setter
    def mesh_loader(self, value):
        self._mesh_loader = value

    @property
    def image_size(self):
        return self._mesh_loader.image_size

    @property
    def image_renderer(self):
        return self._mesh_loader.renderer

    @property
    def target_mesh(self):
        return self._target_mesh

    @target_mesh.setter
    def target_mesh(self, mesh):
        self._target_mesh = scale_and_normalise(mesh)

    @property
    def default_camera(self):
        return self._default_camera

    @default_camera.setter
    def default_camera(self, value):
        self._default_camera = value
        self.renderer_textured.rasterizer.cameras = value
        self.renderer_textured.shader.cameras = value

    @property
    def random_views(self):
        return np.random.permutation(self.num_views).tolist()[:self.num_views_per_iteration]
