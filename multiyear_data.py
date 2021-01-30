import logging
import os
import pathlib
import pvlib
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import rdtools
import seaborn as sns

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

os.environ['NUMEXPR_MAX_THREADS'] = '30'
LOGGER.debug('NumExpr max threads: %s', os.getenv('NUMEXPR_MAX_THREADS'))

sns.set()
plt.ion()

# no inverters, just DC power
# INVERTERS = pvlib.pvsystem.retrieve_sam('CECInverter')
# INVERTER_10K = INVERTERS['SMA_America__SB10000TL_US__240V_']

# choose typical front-contact silicon modules
CECMODS = pvlib.pvsystem.retrieve_sam('CECMod')
CECMOD_POLY = CECMODS['Canadian_Solar_Inc__CS6X_300P']
CECMOD_MONO = CECMODS['Canadian_Solar_Inc__CS6X_300M']

NREL_API_KEY = os.getenv('NREL_API_KEY', 'DEMO_KEY')
EMAIL = os.getenv('EMAIL', 'bwana.marko@yahoo.com')

# read years from SURFRAD path
PATH = pathlib.Path('C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/Bondville_IL')
YEARS = list(str(y) for y in PATH.iterdir())
EDAILY = []

for year in YEARS:
    LOGGER.debug('year: %s', year)
    yearpath = PATH / year
    #header, data = pvlib.iotools.get_psm3(LATITUDE, LONGITUDE, NREL_API_KEY, EMAIL, names=str(year))
    data = [pvlib.iotools.read_surfrad(f) for f in yearpath.iterdir()]
    dfs, heads = zip(*data)
    df = pd.concat(dfs)
    header = heads[0]

    # get solar position
    LATITUDE = header['latitude']
    LONGITUDE = header['longitude']
    ELEVATION = header['elevation']

    TIMES = df.index
    sp = pvlib.solarposition.get_solarposition(
            TIMES, LATITUDE, LONGITUDE)
    solar_zenith = sp.apparent_zenith.values
    solar_azimuth = sp.azimuth.values
    zenith = sp.zenith.values
    ghi = df.ghi.values

    # we don't trust SURFRAD DNI or DHI b/c it's often missing
    # dni = df.dni.values
    # dhi = df.dhi.values

    # we also don't trust air temp, RH, or pressure, same reason
    # temp_air = df.temp_air.values
    # relative_humidity = df.relative_humidity.values
    # pressure = df.pressues.values
    # wind_speed = df.wind_speed.values

    # check the calculated zenith from SURFRAD
    ze_mbe = 100 * (
        sum(solar_zenith - df.solar_zenith.values)
        / sum(df.solar_zenith.values))
    LOGGER.debug(f'zenith MBE: {ze_mbe}%')

    # get irrad components
    irrad = pvlib.irradiance.erbs(ghi, zenith, TIMES)
    dni = irrad.dni.values
    dhi = irrad.dhi.values
    kt = irrad.kt.values  # clearness index

    # calculate irradiance inputs
    dni_extra = pvlib.irradiance.get_extra_radiation(TIMES).values

    # estimate air temp
    TL = pvlib.clearsky.lookup_linke_turbidity(TIMES, LATITUDE, LONGITUDE)
    AM = pvlib.atmosphere.get_relative_airmass(solar_zenith)
    PRESS = pvlib.atmosphere.alt2pres(ELEVATION)
    AMA = pvlib.atmosphere.get_absolute_airmass(AM, PRESS)
    CS = pvlib.clearsky.ineichen(solar_zenith, AM, TL, ELEVATION, dni_extra)
    cs_temp_air = rdtools.clearsky_temperature.get_clearsky_tamb(
        TIMES, LATITUDE, LONGITUDE)
    temp_air = cs_temp_air * ghi / CS.ghi.values

    # tracker positions
    tracker = pvlib.tracking.singleaxis(solar_zenith, solar_azimuth)
    surface_tilt = tracker['surface_tilt']
    surface_azimuth = tracker['surface_azimuth']

    # irrad components in plane of array
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, dni_extra=dni_extra, model='haydavies')
    aoi = tracker['aoi']
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
            surface_tilt, ghi)
    poa = pvlib.irradiance.poa_components(
            aoi, dni, poa_sky_diffuse, poa_ground_diffuse)
    poa_direct = poa['poa_direct']
    poa_diffuse = poa['poa_diffuse']
    poa_global = poa['poa_global']
    iam = pvlib.iam.ashrae(aoi)
    effective_irradiance = poa_direct*iam + poa_diffuse
    temp_cell = pvlib.temperature.pvsyst_cell(poa_global, temp_air)

    # this is the magic
    cecparams = pvlib.pvsystem.calcparams_cec(
            effective_irradiance, temp_cell,
            CECMOD_MONO.alpha_sc, CECMOD_MONO.a_ref,
            CECMOD_MONO.I_L_ref, CECMOD_MONO.I_o_ref,
            CECMOD_MONO.R_sh_ref, CECMOD_MONO.R_s, CECMOD_MONO.Adjust)
    mpp = pvlib.pvsystem.max_power_point(*cecparams, method='newton')
    mpp = pd.DataFrame(mpp, index=TIMES)
    Edaily = mpp.p_mp.resample('D').sum()
    EDAILY.append(Edaily)

yield_daily = pd.concat(EDAILY) / 300.0 / 24 * 100
yield_daily.plot()
plt.ylabel('Daily DC Capacity [%]')
plt.title('Multiyear data')
