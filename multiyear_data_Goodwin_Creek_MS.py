import calendar
import json
import logging
import os
import pathlib
import warnings
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import rdtools
from scipy import stats
import seaborn as sns
import psm3tmy

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
# for now, for simplicity, just one mono-silicon module, mono
CECMOD_MONO = CECMODS['Canadian_Solar_Inc__CS6X_300M']

NREL_API_KEY = os.getenv('NREL_API_KEY', 'DEMO_KEY')
EMAIL = os.getenv('EMAIL', 'bwana.marko@yahoo.com')

# read years from SURFRAD path
PATH = pathlib.Path(
    'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/Goodwin_Creek_MS')
YEARS = list(str(y) for y in PATH.iterdir())

TMY2_PATH = 'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/13893.tm2'
TMY3_PATH = 'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/722359TYA.CSV'
TMY3_PATH_2 = 'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/722356TYA.CSV'
TMY3_PATH_3 = 'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/723340TYA.CSV'
TMY3_PATH_4 = 'C:/Users/SFValidation3/Desktop/Mark Mikofski/SURFRAD/723306TYA.CSV'

# accumulate daily energy
EDAILY = {}

for year in YEARS:
    LOGGER.debug('year: %s', year)
    yearpath = PATH / year
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

    # zero out negative (night?) GHI
    ghi = np.where(ghi < 0, 0, ghi)

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
    year_start = year.rsplit('\\', 1)[1]
    year_minutes = pd.date_range(
        start=year_start, freq='T', periods=527040, tz='UTC')
    TL = pvlib.clearsky.lookup_linke_turbidity(TIMES, LATITUDE, LONGITUDE)
    AM = pvlib.atmosphere.get_relative_airmass(solar_zenith)
    PRESS = pvlib.atmosphere.alt2pres(ELEVATION)
    AMA = pvlib.atmosphere.get_absolute_airmass(AM, PRESS)
    CS = pvlib.clearsky.ineichen(solar_zenith, AMA, TL, ELEVATION, dni_extra)
    cs_temp_air = rdtools.clearsky_temperature.get_clearsky_tamb(
        year_minutes, LATITUDE, LONGITUDE)

    # scale clear sky air temp to SURFRAD, and zero out night
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_air = np.where(
            CS.ghi.values <= 0, 0,
            cs_temp_air.loc[TIMES].values * ghi / CS.ghi.values)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cecparams = pvlib.pvsystem.calcparams_cec(
                effective_irradiance, temp_cell,
                CECMOD_MONO.alpha_sc, CECMOD_MONO.a_ref,
                CECMOD_MONO.I_L_ref, CECMOD_MONO.I_o_ref,
                CECMOD_MONO.R_sh_ref, CECMOD_MONO.R_s, CECMOD_MONO.Adjust)
        mpp = pvlib.pvsystem.max_power_point(*cecparams, method='newton')
    mpp = pd.DataFrame(mpp, index=TIMES)

    # find ourly averages and daily totals
    Ehourly = mpp.p_mp.resample('H').mean()
    Edaily = Ehourly.resample('D').sum()
    LOGGER.debug('%s annual energy: %g[kWh]', year_start, sum(Edaily) / 1000.0)
    EDAILY[year_start] = Edaily

# get yearly totals scaled to number of days
leapdays = 366.0 if calendar.isleap(int(year_start)) else 365.0
EYEAR = {
    y: sum(e)/1000.0 * (leapdays/len(e)) for y, e in EDAILY.items()
    if len(e) > 350}
EYEAR = pd.Series(EYEAR)
LOGGER.info('annual statistics [kWh]:\n%r', EYEAR.describe())
P50 = EYEAR.median()
P90 = EYEAR.quantile(0.1)

# run PSM3 TMY
psm3edaily = psm3tmy.run_psm3tmy(LATITUDE, LONGITUDE, CECMOD_MONO)
psm3eyear = sum(psm3edaily) / 1000.0
LOGGER.debug('PSM3: %g[kWh]', psm3eyear)
psm3quantile = stats.percentileofscore(EYEAR, psm3eyear)
LOGGER.debug('%g quantile of SURFRAD years', psm3quantile)

# run TMY2
tmy2edaily = psm3tmy.run_tmy2(TMY2_PATH, LATITUDE, LONGITUDE, CECMOD_MONO)
tmy2eyear = sum(tmy2edaily) / 1000.0
LOGGER.debug('TMY2 (%s): %g[kWh]', TMY2_PATH, tmy2eyear)
tmy2quantile = stats.percentileofscore(EYEAR, tmy2eyear)
LOGGER.debug('%g quantile of SURFRAD years', tmy2quantile)

# run TMY3
tmy3edaily = psm3tmy.run_tmy3(TMY3_PATH, LATITUDE, LONGITUDE, CECMOD_MONO)
tmy3eyear = sum(tmy3edaily) / 1000.0
LOGGER.debug('TMY3 (%s): %g[kWh]', TMY3_PATH, tmy3eyear)
tmy3quantile = stats.percentileofscore(EYEAR, tmy3eyear)
LOGGER.debug('%g quantile of SURFRAD years', tmy3quantile)
# run TMY3 #2 GREENVILLE
tmy3edaily_2 = psm3tmy.run_tmy3(TMY3_PATH_2, LATITUDE, LONGITUDE, CECMOD_MONO)
tmy3eyear_2 = sum(tmy3edaily_2) / 1000.0
LOGGER.debug('TMY3 (%s): %g[kWh]', TMY3_PATH_2, tmy3eyear_2)
tmy3quantile_2 = stats.percentileofscore(EYEAR, tmy3eyear_2)
LOGGER.debug('%g quantile of SURFRAD years', tmy3quantile_2)
# run TMY3 #3 MEMPHIS INTL AP
tmy3edaily_3 = psm3tmy.run_tmy3(TMY3_PATH_3, LATITUDE, LONGITUDE, CECMOD_MONO)
tmy3eyear_3 = sum(tmy3edaily_3) / 1000.0
LOGGER.debug('TMY3 (%s): %g[kWh]', TMY3_PATH_3, tmy3eyear_3)
tmy3quantile_3 = stats.percentileofscore(EYEAR, tmy3eyear_3)
LOGGER.debug('%g quantile of SURFRAD years', tmy3quantile_3)
# run TMY3 #4 COLUMBUS AFB
tmy3edaily_4 = psm3tmy.run_tmy3(TMY3_PATH_4, LATITUDE, LONGITUDE, CECMOD_MONO)
tmy3eyear_4 = sum(tmy3edaily_4) / 1000.0
LOGGER.debug('TMY3 (%s): %g[kWh]', TMY3_PATH_4, tmy3eyear_4)
tmy3quantile_4 = stats.percentileofscore(EYEAR, tmy3eyear_4)
LOGGER.debug('%g quantile of SURFRAD years', tmy3quantile_4)

# stop logging
LOGGER.setLevel(logging.CRITICAL)

# site name:
SITE = PATH.parts[-1]

# make plots
f, ax = plt.subplots(2, 1, figsize=(10, 8), num=SITE)
yield_daily = pd.concat(EDAILY.values()) / 300.0 / 24.0 * 100.0
yield_daily.plot(ax=ax[0], label='daily')
yield_yearly = EYEAR*1000 / 300.0 / 8760.0 * 100.0
yield_yearly.index = pd.date_range(start=f'{EYEAR.index[0]}', end=f'{int(EYEAR.index[-1])+1}',freq='Y')
yield_yearly.plot(ax=ax[0], label='yearly')
ax[0].set_ylabel('Daily DC Capacity [%]')
ax[0].set_title(f'{SITE} Multiyear Data')
sns.histplot(EYEAR.values, kde=True, ax=ax[1])
ylim = ax[1].get_ylim()
ax[1].plot([P50, P50], ylim, 'b--', [P90, P90], ylim, 'b--')
ax[1].plot([psm3eyear]*2, ylim)
ax[1].plot([tmy2eyear]*2, ylim, 'g--')
ax[1].annotate(f' {tmy2quantile:g}%', (tmy2eyear, 0.95*ylim[1]))
ax[1].plot([tmy3eyear]*2, ylim, 'k--')
ax[1].annotate(f' {tmy3quantile:g}%', (tmy3eyear, 0.85*ylim[1]))
ax[1].plot([tmy3eyear_2]*2, ylim, 'k--')
ax[1].annotate(f' {tmy3quantile_2:g}%', (tmy3eyear_2, 0.75*ylim[1]))
ax[1].plot([tmy3eyear_3]*2, ylim, 'k--')
ax[1].annotate(f' {tmy3quantile_3:g}%', (tmy3eyear_3, 0.65*ylim[1]))
ax[1].plot([tmy3eyear_4]*2, ylim, 'k--')
ax[1].annotate(f' {tmy3quantile_4:g}%', (tmy3eyear_4, 0.55*ylim[1]))
ax[1].legend(['KDE', 'P50', 'P90', 'PSM3', 'TMY2', 'TMY3'])
ax[1].set_title(
    f'{SITE} Distribution: P50 = {P50:g}[kWh], P90 = {P90:g}[kWh]'
    f'\nPSM3 = {psm3eyear:g}[kWh], TMY2 = {tmy2eyear:g}[kWh], TMY3(722359) = {tmy3eyear:g}[kWh]'
    f'\nTMY3(722356) = {tmy3eyear_2:g}[kWh], TMY3(723340) = {tmy3eyear_3:g}[kWh], TMY3(723306) = {tmy3eyear_4:g}[kWh]')
plt.tight_layout()

# save all
plt.savefig(f'{SITE}.png')
yield_daily.to_csv(f'{SITE}_yield_daily.csv', index_label='datetime')
EYEAR.to_csv(f'{SITE}_EYEAR.csv', index_label='year', header=['EYEAR'])
results = {
    'P50': P50, 'P90': P90,
    'psm3eyear': psm3eyear, 'psm3quantile': psm3quantile,
    'tmy2eyear': tmy2eyear, 'tmy2quantile': tmy2quantile,
    'tmy3eyear': tmy3eyear, 'tmy3quantile': tmy3quantile,
    'tmy3eyear_2': tmy3eyear_2, 'tmy3quantile_2': tmy3quantile_2,
    'tmy3eyear_3': tmy3eyear_3, 'tmy3quantile_3': tmy3quantile_3,
    'tmy3eyear_4': tmy3eyear_4, 'tmy3quantile_4': tmy3quantile_4,
}
with open(f'{SITE}_results.json', 'w') as f:
    json.dump(results, f, indent=2)
