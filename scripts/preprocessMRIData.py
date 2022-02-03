import sys
import numpy as np
import cv2
import pywt
import scipy
from nilearn import image

def construct_linear_operator(phi):
    n = phi.shape[0]
    operator = np.zeros((n**2, n**2), complex)
    for m in range(n**2):
        i = int(m / n)
        j = m % n
        for l in range(n**2):
            index_1 = l % n
            index_2 = int(l / n)
            operator[m, l] = phi[index_1, j] * phi[index_2, i]

    return operator

def construct_FT_operator(dim):
    dim = new_dim
    FT_mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        basis_vec = np.zeros(dim)
        basis_vec[i] = 1
        basis_ft = np.fft.fft(basis_vec)
        FT_mat[:, i] = basis_ft

    FT_mat_large = construct_linear_operator(FT_mat.T)

    return FT_mat_large

def construct_DWT_operator(dim):
    wavelet_mat2 = np.zeros((dim, dim), complex)
    for i in range(dim):
        basis_vec = np.zeros(dim)
        basis_vec[i] = 1
        basis_cA, basis_cB = pywt.dwt(basis_vec, 'sym2', 'smooth')
        wavelet_mat2[:, i] = np.concatenate((basis_cA[:-1], basis_cB[:-1]))
    temp = np.linalg.svd(wavelet_mat2)
    psi_mat = temp[0]
    psi_mat_large = np.real(construct_linear_operator(psi_mat))

    return psi_mat_large

new_dim = int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3] + str(new_dim)

slices = np.arange(30, 171, 10)

FT_mat_large = construct_FT_operator(new_dim)
psi_mat_large = construct_DWT_operator(new_dim)

np.save(output_path + '/FT_mat', FT_mat_large)
np.save(output_path + '/basis_mat', psi_mat_large)

img = image.load_img(input_path)
img_array = img.get_fdata()
res = cv2.resize(img_array, dsize=(new_dim, new_dim),
                 interpolation=cv2.INTER_CUBIC)

for slice_index in slices:
    this_slice = res[:, :, slice_index]
    flattened_ft = np.fft.fft2(this_slice).ravel()
    np.save(output_path + '/FT_vec_' + str(slice_index), flattened_ft)
    np.save(output_path + '/image_' + str(slice_index), this_slice.ravel())
