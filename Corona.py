# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 00:09:16 2022

@author: juliu
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from astropy.convolution import AiryDisk2DKernel, convolve_fft
from tqdm import tqdm

def Create_Source(Object, FOV, L):
    if Object == 'Moon':
        MoonSize = 0.5 # deg
        SourceFunction = np.zeros((L,L))

        Y, X = np.ogrid[:L, :L]
        dist_from_center = np.sqrt((X-(L-1)/2+1)**2 + (Y-(L-1)/2+1)**2)
        mask = dist_from_center <= MoonSize*L/FOV/2 # Diameter of Moon is 0.5 deg

        SourceFunction[mask] = 1
    else:
        print("Object not in the list!")
    return SourceFunction


def wavelength_to_radius(PS , # particle size in micron 
                         Lambda # Incident light wavelength
                         ):
    theta = 250/PS*0.5/2*(Lambda/600) # Out put is radius
    return theta # deg

def Monochromatic_Image(Lambda, PS, SourceFunction, Intensity):
    Airy = AiryDisk2DKernel(wavelength_to_radius(PS, Lambda))
    return convolve_fft(SourceFunction, Airy)*Intensity

def Integrated_Image(wavelength, PS, SourceFunction, Spectrum):
    Arr = np.zeros(np.shape(SourceFunction))
    dlambda = (wavelength[-1] - wavelength[0])/np.shape(wavelength)[0]
    for i, Lambda in tqdm(enumerate(wavelength)):
        Arr += dlambda*Monochromatic_Image(Lambda, PS, SourceFunction, Spectrum[i])
    return Arr

def planck(wav, T):
    h = 6.626e-34
    c = 299792458
    k = 1.38e-23
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    Spectrum = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return Spectrum



def auto_adjust(img):
    Max = np.max(img)
    Min = np.min(img)
    #img -= Min
    img = (img - np.ones(np.shape(img))*Min)/Max
    img = exposure.adjust_gamma(img, 1/5)
    return img


def Sep_frame_mode(img):
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(SourceFunction)
    axes[1,0].imshow(img[:,:,2], cmap='Reds')
    axes[0,1].imshow(img[:,:,1], cmap='Greens')
    axes[1,1].imshow(img[:,:,0], cmap='Blues')
    for ax in axes:
        ax.axis("off")
    plt.show()
    
def One_frame_mode(img, save=True):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.axis("off")
    plt.subplots_adjust(0,0,1,1)
    #plt.tight_layout()
    if save:
        plt.savefig("Corona.jpg", dpi=300)
        print("Save!")
    else:
        plt.show()
    
def main():
    FOV = 15 # Degree
    MoonSize = 0.5 # Degree
    L = 501 # Pixels
    cloud_size = 1 # micron
    img = np.zeros((L,L,3))
    SourceFunction = Create_Source('Moon', FOV, L)
    # https://en.wikipedia.org/wiki/Spectral_sensitivity#/media/File:Cones_SMJ2_E.svg
    wave_B = np.linspace(420, 475, 10)
    wave_G = np.linspace(510, 590, 10)
    wave_R = np.linspace(620, 700, 10)
    B = Integrated_Image(wave_B, cloud_size, SourceFunction, planck(wave_B, 5500))
    G = Integrated_Image(wave_G, cloud_size, SourceFunction, planck(wave_G, 5500))
    R = Integrated_Image(wave_R, cloud_size, SourceFunction, planck(wave_R, 5500))
    img[:,:,0] = B
    img[:,:,1] = G
    img[:,:,2] = R
    img = auto_adjust(img)
    #plt.hist(img.ravel())
    #plt.show()
    One_frame_mode(img)

if __name__ == '__main__':
    main()
