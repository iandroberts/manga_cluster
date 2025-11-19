from pathlib import Path
from requests.exceptions import HTTPError
import sys

from astropy.table import Table
import astropy.units as u
from marvin.core.exceptions import MarvinError
from marvin.tools import Cube, Maps
from marvin.tools.query import Query
from marvin.utils.general.general import get_drpall_table
import matplotlib.pyplot as plt
import numpy as np
from specutils import Spectrum
from specutils.manipulation import LinearInterpolatedResampler
from tqdm import trange, tqdm

def compile_sample():
    home = Path.home()
    drppath = f"{str(home)}/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
    drpall = Table.read(drppath)
    primary = drpall['mngtarg1'] & 2**10
    secondary = drpall['mngtarg1'] & 2**11
    color_enhanced = drpall['mngtarg1'] & 2**12

    main_sample = np.logical_or.reduce((primary, secondary, color_enhanced))
    selection = np.logical_and.reduce((
        main_sample,
        drpall["z"] < 0.1,
    ))
    plateifu = drpall["plateifu"][selection]
    return plateifu

def reproject_spectral_axis(wav, flux):
    input_spec = Spectrum(spectral_axis=wav, flux=flux)
    linear = LinearInterpolatedResampler()
    new_wav = np.arange(3700, 9401)*u.AA
    new_spec = linear(input_spec, new_wav)
    return new_spec.spectral_axis, new_spec.flux

def preprocess_individual_cube(plateifu, lmin=3695, lmax=9405):
    cube = Cube(plateifu=f"{plateifu}")
    z = cube.nsa["z"]
    maps = Maps(plateifu=f"{plateifu}")
    good = cube.flux.mask == 0
    rmap = maps.spx_ellcoo_r_re
    spectra = radial_stack(cube.flux.value*good, rmap.value)
    wavrest = cube.flux.wavelength.value / (1+z)
    wavmask = np.logical_and.reduce((
        wavrest >= lmin,
        wavrest <= lmax,
    ))
    new_specs = []
    for i in range(spectra.shape[0]):
        new_wav, new_flux = reproject_spectral_axis(wavrest[wavmask]*u.AA,
            spectra[i, wavmask]*1e-17*(u.erg/u.s/u.cm**2/u.AA))
        new_specs.append(new_flux.value*1e+17)
    return new_wav.value, np.array(new_specs)

def radial_stack(flux, rmap):
    radii = np.arange(0, 1.51, 0.25)
    spectra = np.zeros((radii.size-1, flux.shape[0]))
    for i in range(radii.size-1):
        mask = np.logical_and.reduce((
            rmap >= radii[i],
            rmap < radii[i+1],
        ))
        spectra[i] = np.nansum(mask[np.newaxis, :, :] * flux, axis=(1,2))
    return spectra

def preprocess_all_cubes(force=False):
    plateifus = compile_sample()
    for plateifu in tqdm(plateifus):
        specpath = Path(f"preprocessed_spectra/{plateifu}.fits")
        if specpath.is_file() and (not force):
            print(f"Skipping {plateifu}, file already exists")
            continue
        try:
            wav, specs = preprocess_individual_cube(plateifu)
            write_preprocessed_specs(f"{plateifu}", wav, specs)
        except (MarvinError, TimeoutError):
            continue

def write_preprocessed_specs(plateifu, wav, flux):
    dirpath = Path("preprocessed_spectra")
    dirpath.mkdir(parents=True, exist_ok=True)
    tout = Table()
    tout["wav"] = wav
    for i in range(flux.shape[0]):
        tout[f"flux{i+1}"] = flux[i, :]
    tout.write(f"{str(dirpath)}/{plateifu}.fits", overwrite=True)

if __name__ == "__main__":
    preprocess_all_cubes()
