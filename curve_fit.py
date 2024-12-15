import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.ndimage import convolve
from scipy.ndimage import correlate
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
import os
from pyiqa.archs import gmsd_arch
import torch
from my_ssim import SSIM as ssim
from my_ssim import MSSIM as mssim

def load_gray_img(image_path):
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image
    # return image

class LogisticCurve:
    # Variance-weighted Regression: weights = [1/MOS[i].variance]
    def __init__(self, X_data, Y_data, weights=None):
        self.X_data = X_data
        self.Y_data = Y_data
        self.weights = weights
        
        self.params, _ = self.optimize_curve()
        self.x_fit = np.linspace(min(X_data), max(X_data))
        self.y_fit = self.logistic(self.x_fit, *self.params)
        
    def optimize_curve(self):
        initial_params = [np.max(self.Y_data), 1, np.median(self.X_data), np.min(self.Y_data)]
        sigma = None if self.weights is None else 1 / np.sqrt(self.weights)
        params, _ = curve_fit(self.logistic, self.X_data, self.Y_data, p0=initial_params, sigma=sigma, maxfev=10000)
        return params, _
        
    def logistic(self, x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def getX(self):
        return self.x_fit

    def getY(self):
        return self.y_fit

    # Correlation Coefficient
    def getCC(self):
        return np.corrcoef(self.X_data, self.Y_data)[0, 1]

    # Mean Absolute Error
    def getMAE(self):
        y_pred = self.logistic(self.X_data, *self.params)
        return np.mean(np.abs(self.Y_data - y_pred))

    # Root mean squared
    def getRMS(self):
        y_pred = self.logistic(self.X_data, *self.params)
        return np.sqrt(np.mean((self.Y_data - y_pred) ** 2))

    # Outlier Ratio
    def getOR(self, threshold=2): # Paper's deviation threshold
        y_pred = self.logistic(self.X_data, *self.params)
        error = np.abs(self.Y_data - y_pred)
        outliers = np.sum(error > threshold * np.std(self.Y_data))
        return outliers / len(self.Y_data)

    def getSpearman(self):
        spearman_coefficient, _ = spearmanr(self.X_data, self.Y_data)
        return spearman_coefficient

    def getPerformance(self, threshold=1, as_dataframe=False):
        performance = {
            "CC": self.getCC(),
            "MAE": self.getMAE(),
            "RMS": self.getRMS(),
            "OR": self.getOR(threshold),
            "SROCC": self.getSpearman()
        }
        if as_dataframe:
            return pd.DataFrame([performance])  # Convert to DataFrame with one row
        return performance