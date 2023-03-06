import numpy as np
import cv2
import os
from scipy import signal
from scipy import misc
from generate_PSF import PSF
from generate_trajectory import Trajectory
import os,cv2,csv,sys, glob,math


class BlurImage(object):

    def __init__(self, imag, PSFs=None, part=None):
    
        self.original = imag#cv2.imread(self.image_path,-1) #misc.imread(self.image_path)
        self.shape = self.original.shape
        if len(self.shape) < 3:
            raise Exception('We support only RGB images yet.')
        elif self.shape[0] != self.shape[1]:
            raise Exception('We support only square images yet.')

        self.PSFs = PSFs
        self.part = part
        self.result = []

    def blur_image(self, save=False):

        psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
            blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.abs(blured))

        return cv2.cvtColor(result[0] * 255, cv2.COLOR_RGB2BGR)