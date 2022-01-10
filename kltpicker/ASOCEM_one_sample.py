import numpy as np
import numpy.linalg as linalg
import cv2
from skimage import measure
import skimage
from scipy.ndimage import gaussian_filter
import warnings
from skimage.morphology import erosion
from skimage.transform import resize
import mrcfile


warnings.filterwarnings("ignore")


class DataHolder:
    def __init__(self, I, area, stop_condition_n, max_iter, contamination_criterion):
        self.stop_condition_n = stop_condition_n
        self.max_iter = max_iter
        self.area = int(area)
        self.cov_mat_size = area * area
        self.contamination_criterion = contamination_criterion

        # Find shape of I and computing phi_0
        self.size_x, self.size_y = I.shape
        self.phi = initialize_phi_0(self.size_x, self.size_y)
        self.old_phi = self.phi.copy()

        # Defining I_patches
        self.I = I
        self.I_patches = np.empty((self.size_x // area, self.size_y // area, self.cov_mat_size))
        self.I_patches_vector = np.empty(((self.size_x // area) * (self.size_y // area), self.cov_mat_size))
        I_patches = self.I_patches
        for x in range(self.size_x // area):
            for y in range(self.size_y // area):
                I_patches[x, y] = self.I[x * area:(x + 1) * area, y * area:(y + 1) * area].flatten()

        self.I_patches_vector = np.reshape(I_patches, ((self.size_x // area) * (self.size_y // area), self.cov_mat_size))

        # Other variables
        self.iteration = 0
        self.mu0_est = np.empty(self.cov_mat_size)
        self.mu1_est = np.empty(self.cov_mat_size)
        self.cov0_inv = np.empty((self.cov_mat_size, self.cov_mat_size))
        self.cov1_inv = np.empty((self.cov_mat_size, self.cov_mat_size))
        self.logdet0 = 0
        self.logdet1 = 0

    def compute_statistics(self):
        phi_max = skimage.measure.block_reduce(self.phi, (self.area, self.area), np.max)
        phi_min = skimage.measure.block_reduce(self.phi, (self.area, self.area), np.min)

        patch_0 = self.I_patches[np.where(phi_max >= 0)].T
        patch_1 = self.I_patches[np.where(phi_min <= 0)].T

        # Compute mean and covariance
        self.mu0_est = np.mean(patch_0, 1)
        cov0_est = np.cov(patch_0)
        self.mu1_est = np.mean(patch_1, 1)
        cov1_est = np.cov(patch_1)

        self.cov0_inv = linalg.pinv(cov0_est)
        self.cov1_inv = linalg.pinv(cov1_est)
        self.logdet0 = logdet_amitay(cov0_est)
        self.logdet1 = logdet_amitay(cov1_est)

    def get_rts(self):
        x0 = self.I_patches_vector - self.mu0_est[np.newaxis, :]
        x1 = self.I_patches_vector - self.mu1_est[np.newaxis, :]
        tmp_logdet = self.logdet1 - self.logdet0
        tmp_qf = multi_quadratic_form(x1, self.cov1_inv) - multi_quadratic_form(x0, self.cov0_inv)
        rts = (tmp_logdet + tmp_qf) / (2 * self.cov_mat_size)
        return np.reshape(rts, (self.size_x // self.area, self.size_y // self.area))

    def step(self, nu, dt, eps):
        rts = -nu + self.get_rts()

        dt_deltas = dt * delta_eps(self.phi[::self.area, ::self.area], eps)
        dt_deltas *= rts
        for i in range(self.area):
            for j in range(self.area):
                self.phi[i::self.area, j::self.area] += dt_deltas

    def neumann_bound_cond_mod(self, h=1):
        self.phi[[0, -1, 0, -1], [0, 0, -1, -1]] = self.phi[[h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
        self.phi[[0, -1], 1:-1] = self.phi[[h, -h - 1], 1:-1]
        self.phi[1:-1, [0, -1]] = self.phi[1:-1, [h, -h - 1]]

    def check_stop_criterion(self, tol):
        if self.iteration % self.stop_condition_n == 0:
            area_new = self.phi > 0
            area_old = self.old_phi > 0
            changed = area_new != area_old
            if np.count_nonzero(changed) / np.count_nonzero(area_old) < tol or self.iteration >= self.max_iter:
                return 1
            self.old_phi = self.phi.copy()
        return 0

    def get_phi(self):
        phi = self.phi.copy()
        I = self.I
        if self.contamination_criterion == 'size':
            def est_func(group):
                return np.count_nonzero(group)
        elif self.contamination_criterion == 'power':
            def est_func(group):
                return np.mean(I[group])
        else:
            raise ValueError('contamination_criterion can only be size or power, got {} instead'.
                             format(self.contamination_criterion))

        group1_score = est_func(phi > 0)
        group2_score = est_func(phi < 0)
        if group1_score >= group2_score:
            phi *= -1
        return phi


def ASOCEM_ver1(micrograph, particle_size, downscale_size, area_size, contamination_criterion, out_size, save_path=None):

    # Require odd area_size, and odd downscale_size such that area_size | downscale_size
    area_size = area_size - 1 if area_size % 2 == 0 else area_size

    downscale_size_max = downscale_size

    while downscale_size_max % area_size != 0 or downscale_size_max % 2 == 0:
        downscale_size_max -= 1

    scalingSz = downscale_size_max / max(micrograph.shape)
    downscale_size_min = np.floor(scalingSz * min(micrograph.shape))

    while downscale_size_min % area_size != 0 or downscale_size_min % 2 == 0:
        downscale_size_min -= 1

    if micrograph.shape[0] >= micrograph.shape[1]:
        size_x, size_y = int(downscale_size_max), int(downscale_size_min)
    else:
        size_x, size_y = int(downscale_size_min), int(downscale_size_max)

    # De-noising filter
    sigma = 1.
    I = gaussian_filter(micrograph, sigma, mode='nearest', truncate=np.ceil(2 * sigma) * sigma)

    # Rescaling
    I = cryo_downsample(I, (size_x, size_y))

    # Executing ASOCEM
    phi = ASOCEM(I, area_size, contamination_criterion)

    # Post processing
    scaling_size = downscale_size / max(micrograph.shape)
    d = max(3, int(np.ceil(scaling_size * particle_size / 8)))
    phi[:d] = 0
    phi[-d:] = 0
    phi[:, :d] = 0
    phi[:, -d:] = 0

    phi_seg = np.zeros(phi.shape)
    se_erod = max(area_size, int(np.ceil(scaling_size * particle_size / 6)))
    phi_erod = erosion((phi > 0).astype('float'), np.ones((se_erod, se_erod)))
    connected_components = measure.label(phi_erod)

    group_threshold = (scaling_size * particle_size) ** 2
    group_ids, group_sizes = np.unique(connected_components, return_counts=True)
    indices = np.argsort(-group_sizes)
    group_ids, group_sizes = group_ids[indices], group_sizes[indices]
    tmp = np.full(connected_components.shape, False)
    for group_id, group_size in zip(group_ids, group_sizes):
        if np.sum(phi_erod[connected_components == group_id]) == 0:
            continue
        tmp = np.logical_or(tmp, connected_components == group_id)
        if group_size > group_threshold:
            phi_seg[connected_components == group_id] = 1

    phi_seg = cv2.dilate(phi_seg, np.ones((se_erod, se_erod)))
    # This resize is the closest I could get to matlab, still minor error
    phi_seg_big = resize(phi_seg, out_size, order=0)

    if save_path is not None:
        save_phi_seg = resize(phi_seg, micrograph.shape, order=0)

        with mrcfile.new(save_path, overwrite=True) as mrc_fh:
            mrc_fh.set_data(save_phi_seg.astype('float32').T)

    return phi_seg_big


def ASOCEM(I, area, contamination_criterion):
    dt = 10 ** 0
    nu = 0
    eps = 1
    tol = 10 ** -3
    max_iter = 300
    area = int(area)

    # Chan Vesse time process
    return chan_vesse_process(I, area, dt, nu, eps, max_iter, tol, contamination_criterion)


def chan_vesse_process(I, area, dt, nu, eps, max_iter, tol, contamination_criterion):
    stop_condition_n = 5
    data_holder = DataHolder(I, area, stop_condition_n, max_iter, contamination_criterion)

    while True:
        # Updating iterations
        data_holder.iteration += 1

        # Computing statistics
        data_holder.compute_statistics()

        # Step
        data_holder.step(nu, dt, eps)

        # Neumann bound
        data_holder.neumann_bound_cond_mod()

        # Stop criteria
        if data_holder.check_stop_criterion(tol):
            break

    return data_holder.get_phi()


def initialize_phi_0(size_x, size_y):
    x, y = np.meshgrid(np.arange(-(size_y // 2), size_y // 2 + 1), np.arange(-(size_x // 2), size_x // 2 + 1))
    phi_0 = (min(size_x, size_y) / 3) ** 2 - (x ** 2 + y ** 2)
    phi_0 /= np.max(np.abs(phi_0))
    return neumann_bound_cond_mod(phi_0)


def delta_eps(t, eps):
    return eps / (np.pi * (eps ** 2 + t ** 2))


def multi_quadratic_form(x, A):
    return np.sum(x @ A * x, -1)


def neumann_bound_cond_mod(f, h=1):
    g = f.copy()
    g[[0, -1, 0, -1], [0, 0, -1, -1]] = g[[h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
    g[[0, -1], 1:-1] = g[[h, -h - 1], 1:-1]
    g[1:-1, [0, -1]] = g[1:-1, [h, -h - 1]]
    return g


def logdet_amitay(mat):
    eig_vals, _ = linalg.eig(mat)
    eig_vals[eig_vals < 10e-8] = 1
    return np.sum(np.log(eig_vals))


def cryo_downsample(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: downsampled x
    """
    dtype_in = x.dtype
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    fourier_dims = np.array([i for i, s in enumerate(out_shape) if 0 < s < in_shape[i]])
    size_in = np.prod(in_shape[fourier_dims])
    size_out = np.prod(out_shape[fourier_dims])

    fx = crop(np.fft.fftshift(np.fft.fft2(x, axes=fourier_dims), axes=fourier_dims), out_shape)
    out = np.fft.ifft2(np.fft.ifftshift(fx, axes=fourier_dims), axes=fourier_dims) * (size_out / size_in)
    return out.astype(dtype_in)


def crop(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out
