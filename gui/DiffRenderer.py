import torch

from typing import Tuple, List
from tqdm import tqdm
import numpy as np

from pytorch3d.renderer import look_at_view_transform, \
    FoVPerspectiveCameras, PointLights, TexturesVertex
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

from .ImageRenderer import ImageRendererStatic


def scale_and_normalise(mesh):
    verts = mesh.verts_packed()

    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])

    new_mesh = mesh.offset_verts(-center)
    new_mesh.scale_verts_((1.0 / float(scale)))

    return new_mesh


class DiffRenderer(object):
    def __init__(self, render_pane, renderer):
        self.device = renderer.device

        self._render_pane = render_pane

        # meshes
        initial_mesh = render_pane.mesh
        initial_vertices_dim = initial_mesh.verts_packed().shape
        num_vertices = initial_vertices_dim[0]

        self._initial_mesh = initial_mesh
        self._target_mesh = None
        self._target_is_set = False

        # renderer
        self.lights = PointLights(device=self.device,
                                  location=[[0.0, 0.0, -3.0]])
        self.renderer_dynamic = renderer
        self.renderer_static = ImageRendererStatic(render_pane.image_size,
                                                   self.device)

        self.losses = {"rgb": {"weight": 1.0, "values": []},
                       "silhouette": {"weight": 1.0, "values": []},
                       "edge": {"weight": 1.0, "values": []},
                       "normal": {"weight": 0.01, "values": []},
                       "laplacian": {"weight": 1.0, "values": []},}

        # initial values
        self.deform_verts = torch.full(initial_vertices_dim,
                                       0.0,
                                       device=self.device,
                                       requires_grad=True)
        self.sphere_verts_rgb = torch.full([1, num_vertices, 3], 0.5,
                                           device=self.device, requires_grad=True)

        # optimiser
        self.num_views = 20
        self.num_views_per_iteration = 2
        self.num_iter = 100
        self.plot_period = 25
        self.optimiser = torch.optim.SGD([self.deform_verts,
                                          self.sphere_verts_rgb],
                                         lr=1.0, momentum=0.9)

    def set_target_mesh(self, mesh) -> None:
        self._target_mesh = scale_and_normalise(mesh)
        self._target_is_set = True

    def render(self) -> None:
        if not self._target_is_set:
            return

        target_cameras, target_silhouette, target_rgb = self.generate_views()
        self.optimise(target_cameras, target_silhouette, target_rgb)

    def generate_views(self) -> Tuple[List[FoVPerspectiveCameras],
                                      List[torch.Tensor],
                                      List[torch.Tensor]]:
        target_meshes = self._target_mesh.extend(self.num_views)

        elev = torch.linspace(0, 360, self.num_views)
        azim = torch.linspace(-180, 180, self.num_views)
        rots, trans = look_at_view_transform(dist=2.7, elev=elev, azim=azim)

        cameras = FoVPerspectiveCameras(device=self.device,
                                        R=rots, T=trans)

        target_cameras = [FoVPerspectiveCameras(device=self.device,
                                                R=rots[None, i, ...],
                                                T=trans[None, i, ...])
                          for i in range(self.num_views)]

        target_images = self.renderer_dynamic(target_meshes,
                                              cameras=cameras,
                                              lights=self.lights)
        silhouette_images = self.renderer_static(target_meshes,
                                                 silhouette=True,
                                                 cameras=cameras,
                                                 lights=self.lights)

        target_silhouette = [silhouette_images[i, ..., 3]
                             for i in range(self.num_views)]

        target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]

        return target_cameras, target_silhouette, target_rgb

    def optimise(self, target_cameras, target_silhouette, target_rgb) -> None:
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
                # learning -- with grad
                predicted_images = self.renderer_static(deformed_mesh,
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
                self._render_pane.mesh = deformed_mesh

            sum_loss.backward()
            self.optimiser.step()

    @property
    def image_size(self) -> int:
        return self._render_pane.image_size

    @property
    def random_views(self) -> list:
        return np.random.permutation(self.num_views).tolist()[:self.num_views_per_iteration]
