# Atmospheric_Corona
This code is used to simulate atmospheric corona.
There are some details remain to be polished. But current version should be somewhat decent.

Corona forms when sun/moon light is diffracted by the small water droplet in the sky. The resulting PSF should be an Airy disk.
In the code, we first convolve the source function (a disk with diameter of 0.5 degree) with Airy kernal. Then we integrate the resulting monochromatic image over wavelength and eventually create RGB image. We do not really consider the transmission curve of any DSLR and the responce function of the human eyes. Nonetheless, current results looks surprisingly good.

![alt text](https://github.com/CFP106020008/Atmospheric_Corona/blob/main/Corona.jpg)
![alt text](https://github.com/CFP106020008/Atmospheric_Corona/blob/main/profile.jpg)
