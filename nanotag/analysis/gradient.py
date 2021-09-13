from numbers import Number

import numpy as np
import torch
import torch.fft
from torch import nn
from tqdm.auto import tqdm

from nanotag.utils import label_to_index_generator


class GaussianModel(nn.Module):

    def __init__(self, sigma, height, fourier_space=False):
        super().__init__()

        if isinstance(sigma, Number):
            sigma = [sigma]

        if isinstance(height, Number):
            height = [height]

        self.sigma = torch.nn.Parameter(data=torch.tensor(sigma, dtype=torch.float32), requires_grad=True)
        self.height = torch.nn.Parameter(data=torch.tensor(height, dtype=torch.float32), requires_grad=True)

        assert len(self.sigma.shape) == 1
        assert len(self.height.shape) == 1
        assert len(self.sigma) == len(self.height)

        self.fourier_space = fourier_space

    def eval_realspace(self, x):
        xdim = len(x.shape)
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.sigma.device, dtype=torch.float32)

        height = self.height / self.height.sum()

        return (torch.exp(-x[..., None] ** 2 / (2 * self.sigma[(None,) * xdim] ** 2)) * height).sum(-1)

    def eval_fourierspace(self, x):
        xdim = len(x.shape)
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.sigma.device, dtype=torch.float32)

        height = self.height / self.height.sum()

        return (torch.exp(-x[..., None] ** 2 * self.sigma[(None,) * xdim] ** 2 * 2 * np.pi ** 2) *
                height[(None,) * xdim] * self.sigma[(None,) * xdim] ** 2 * 2 * np.pi).sum(-1)

    def __call__(self, x):
        if self.fourier_space:
            return self.eval_fourierspace(x)
        else:
            return self.eval_realspace(x)


class InterpolatedModel(nn.Module):

    def __init__(self, nodes, values):
        super().__init__()
        if not torch.is_tensor(nodes):
            nodes = torch.tensor(nodes, dtype=torch.float32)

        self.nodes = nodes

        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32)

        self.values = torch.nn.Parameter(data=values, requires_grad=True)

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.values.device, dtype=torch.float32)
        return Interp1d()(self.nodes, self.values, x)


class AbstractConstraint:

    def __init__(self, apply_to):
        self._apply_to = apply_to

    @property
    def apply_to(self):
        return self._apply_to


class FixPositions(AbstractConstraint):

    def __init__(self, mask):
        self._mask = mask
        super().__init__(apply_to='gradients')

    def apply(self, model):
        model.positions.grad[self._mask] = 0.


class BoundPositions(AbstractConstraint):

    def __init__(self, centers, bound):
        self._centers = centers
        self._bound = bound
        super().__init__(apply_to='parameters')

    def apply(self, model):
        with torch.no_grad():
            vec = model.positions - self._centers
            d = torch.norm(vec, dim=1)
            norm_vec = vec / (d[:, None] + 1e-6)
            model.positions[:] = self._centers + norm_vec * d[:, None].clamp(0, self._bound)


class FixIntensities(AbstractConstraint):

    def __init__(self, mask):
        try:
            self._mask = np.concatenate(mask)
        except ValueError:
            self._mask = mask

        super().__init__(apply_to='gradients')

    def apply(self, model):
        model.intensities.grad[self._mask] = 0.


class CoupleIntensities(AbstractConstraint):

    def __init__(self, labels, per_image=True):
        self._labels = labels
        self._per_image = per_image
        super().__init__(apply_to='gradients')

    def apply(self, model):
        if self._per_image:
            i = 0
            for l in self._labels:
                for indices in label_to_index_generator(l):
                    indices = np.sort(indices + i)
                    model.intensities.grad[indices] = model.intensities.grad[indices].mean()
                i = i + len(l)
        else:
            for indices in label_to_index_generator(self._labels):
                indices = np.sort(indices)
                model.intensities.grad[indices] = model.intensities.grad[indices].mean()


class ProbeSuperposition(nn.Module):

    def __init__(self, shape, probe_model, positions, intensities=None, margin=0, constraints=None):
        super().__init__()

        if isinstance(positions, list):
            self.image_label = np.hstack([[i] * len(p) for i, p in enumerate(positions)])
            positions = np.vstack(positions)
        else:
            self.image_label = np.zeros(len(positions), dtype=np.int)

        if intensities is None:
            intensities = np.ones(len(positions))

        if isinstance(intensities, list):
            intensities = np.concatenate(intensities)

        assert len(positions) == len(self.image_label)
        assert len(intensities) == len(positions)
        self.probe_model = probe_model

        self.positions = torch.nn.Parameter(data=torch.tensor(positions, dtype=torch.float32), requires_grad=True)
        self.intensities = torch.nn.Parameter(data=torch.tensor(intensities, dtype=torch.float32), requires_grad=True)

        self.shape = shape
        self.margin = margin

        if constraints is None:
            constraints = []

        self.constraints = constraints

        kx = np.fft.fftfreq(shape[-2] + 2 * self.margin)
        ky = np.fft.fftfreq(shape[-1] + 2 * self.margin)
        self.k = torch.tensor(np.sqrt(kx[None] ** 2 + ky[:, None] ** 2), dtype=torch.float32)

    def _get_images(self, remove_margin=True):
        shape = (self.shape[0], self.shape[1] + 2 * self.margin, self.shape[2] + 2 * self.margin)

        array = torch.zeros(shape, device=self.positions.device)
        self.k = self.k.to(self.positions.device)
        self.intensities = self.intensities.to(self.positions.device)

        positions = self.positions + self.margin

        rounded = torch.floor(positions).type(torch.long)
        rows, cols = rounded[:, 1], rounded[:, 0]

        rows = torch.clip(rows, 0, array.shape[-2] - 1)
        cols = torch.clip(cols, 0, array.shape[-1] - 1)

        a = (1 - (positions[:, 1] - rows)) * (1 - (positions[:, 0] - cols)) * self.intensities
        b = (positions[:, 1] - rows) * (1 - (positions[:, 0] - cols)) * self.intensities
        c = (1 - (positions[:, 1] - rows)) * (positions[:, 0] - cols) * self.intensities
        d = (rows - positions[:, 1]) * (cols - positions[:, 0]) * self.intensities

        array[self.image_label, rows, cols] += a
        array[self.image_label, (rows + 1) % shape[-2], cols] += b
        array[self.image_label, rows, (cols + 1) % shape[-1]] += c
        array[self.image_label, (rows + 1) % shape[-2], (cols + 1) % shape[-1]] += d

        array = torch.fft.ifftn(torch.fft.fftn(array, dim=(-2, -1)) * self.probe_model(self.k), dim=(-2, -1))

        if remove_margin:
            return array.real[:, self.margin:-self.margin, self.margin:-self.margin]
        else:
            return array.real

    def forward(self):
        return self._get_images()

    def get_loss(self, target, weights=None):
        assert target.shape == self.shape

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device=self.positions.device)

        prediction = self._get_images()
        losses = ((prediction - target) ** 2)
        if weights is not None:
            losses *= weights
        return losses.sum(axis=(-2, -1))

    def get_positions(self):
        positions = self.positions.detach().cpu().numpy()
        positions_list = []
        for i in label_to_index_generator(self.image_label):
            positions_list.append(positions[np.sort(i)])
        return positions_list

    def get_intensities(self):
        intensities = self.intensities.detach().cpu().numpy()
        intensities_list = []
        for i in label_to_index_generator(self.image_label):
            intensities_list.append(intensities[np.sort(i)])
        return intensities_list

    def get_images(self, remove_margin=True):
        return self._get_images(remove_margin=remove_margin).detach().cpu().numpy()

    def optimize(self, target, optimizers, num_iter, weights=None, pbar=True):

        if weights is not None:
            assert target.shape == weights.shape
            weights = torch.tensor(weights, device=self.positions.device)

        target = torch.tensor(target, device=self.positions.device)

        pbar = tqdm(total=num_iter, disable=not pbar)
        for i in range(num_iter):
            loss = self.get_loss(target, weights)
            sum_loss = loss.sum()

            for optimizer in optimizers:
                optimizer.zero_grad()

            sum_loss.backward()

            for contraint in self.constraints:
                if contraint.apply_to != 'gradients':
                    continue

                contraint.apply(self)

            for optimizer in optimizers:
                optimizer.step()

            for contraint in self.constraints:
                if contraint.apply_to != 'parameters':
                    continue

                contraint.apply(self)

            pbar.update(1)
            pbar.set_postfix({'loss': sum_loss.detach().item()})

        pbar.close()
        return loss
