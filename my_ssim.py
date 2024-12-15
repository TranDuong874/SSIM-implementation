import numpy as np
from skimage import filters
from skimage.feature import canny
import scipy
from scipy.ndimage import convolve
from scipy.ndimage import correlate

def gaussian_window(window_size=11, sigma=1.5):
    """Create a Gaussian kernel."""
    half = window_size // 2
    x, y = np.mgrid[-half:half + 1, -half:half + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()
    
def SSIM(X, Y, K1=0.01, K2=0.03, edge=False): # without gaussian window 
    # SSIM is Universal Quality Index (UQI) when C1 = c2 = 0
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    C4 = C2 / 2
    alpha = 1
    gamma = 1
    beta = 1
    delta = 0.1

    mu_x = X.mean()
    mu_y = Y.mean()


    sigma_x2 = np.sum((X - mu_x) ** 2) / (X.shape[0] * X.shape[1] - 1)
    sigma_y2 = np.sum((Y - mu_y) ** 2) / (Y.shape[0] * Y.shape[1] - 1)
    sigma_xy = np.sum((X - mu_x) * (Y - mu_y)) / (X.shape[0] * X.shape[1] - 1)
    luminance = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    contrast = (2 * np.sqrt(sigma_x2) * np.sqrt(sigma_y2) + C2) / (sigma_x2 + sigma_y2 + C2)
    structure = (sigma_xy + C3) / (np.sqrt(sigma_x2) * np.sqrt(sigma_y2) + C3)


    # Compute edge similarity using Canny edge detector
    if edge:
        edges_X = canny(X)
        edges_Y = canny(Y)
        edge_similarity = (np.sum(edges_X * edges_Y) + C4) / (np.sum(edges_X) + np.sum(edges_Y) + C4)
    else:
        edge_similarity = 1
        delta = 1
        
    SSIM_value = (luminance ** alpha) * (contrast ** gamma) * (structure ** beta) * (edge_similarity ** delta)
    return SSIM_value


def MSSIM(X, Y, K1=0.01, K2=0.03, win_size=11, edge=False, gradient=False):
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    C4 = C2 / 2
    alpha = 1
    gamma = 1
    beta = 1
    delta = 0.1

    g_window = gaussian_window(window_size=win_size)
    mu_x = convolve(X, g_window, mode='reflect')
    mu_y = convolve(Y, g_window, mode='reflect')

    sigma_x2 = convolve((X - mu_x) ** 2, g_window, mode='reflect')
    sigma_y2 = convolve((Y - mu_y) ** 2, g_window, mode='reflect')
    sigma_xy = convolve((X - mu_x) * (Y - mu_y), g_window, mode='reflect')

    luminance = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    contrast = (2 * np.sqrt(sigma_x2) * np.sqrt(sigma_y2) + C2) / (sigma_x2 + sigma_y2 + C2)
    structure = (sigma_xy + C3) / (np.sqrt(sigma_x2) * np.sqrt(sigma_y2) + C3)

    if gradient:
        grad_luminance = np.gradient(luminance, axis=1)  # Gradient with respect to Y
        grad_contrast = np.gradient(contrast, axis=1)
        grad_structure = np.gradient(structure, axis=1)

        # Compute the gradients of each component and combine them
        grad_SSIM = (grad_luminance ** alpha) * (grad_contrast ** gamma) * (grad_structure ** beta)

        if edge:
            beta = 1 - delta
            edges_X = canny(X)
            edges_Y = canny(Y)
            edge_similarity = (np.sum(edges_X * edges_Y) + C4) / (np.sum(edges_X) + np.sum(edges_Y) + C4)

            # No need to apply np.gradient to edge_similarity, it's a scalar
            grad_SSIM *= (edge_similarity ** delta)  # Combine edge similarity directly with SSIM gradient

        return grad_SSIM
    else:
        luminance = np.clip(luminance, 0, 1)
        contrast = np.clip(contrast, 0, 1)
        structure = np.clip(structure, 0, 1)

        if edge:
            beta = 1 - delta
            edges_X = canny(X)
            edges_Y = canny(Y)
            edge_similarity = (np.sum(edges_X * edges_Y) + C4) / (np.sum(edges_X) + np.sum(edges_Y) + C4)
        else:
            edge_similarity = 1
            delta = 1

        SSIM_map = (luminance ** alpha) * (contrast ** gamma) * (structure ** beta) * (edge_similarity ** delta)
        MSSIM = np.mean(SSIM_map)

        return MSSIM
