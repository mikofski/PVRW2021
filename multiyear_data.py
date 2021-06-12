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
from pandas.tseries.frequencies import to_offset
import pvlib
import rdtools
from scipy import stats
import seaborn as sns
import psm3tmy

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

os.environ['NUMEXPR_MAX_THREADS'] = '30'
LOGGER.info('NumExpr max threads: %s', os.getenv('NUMEXPR_MAX_THREADS'))

sns.set()
plt.ion()

# no inverters, just DC power

# choose typical front-contact silicon modules
CECMODS = pvlib.pvsystem.retrieve_sam('CECMod')
# for now, for simplicity, just one mono-silicon module, mono
CECMOD_MONO = CECMODS['Canadian_Solar_Inc__CS6X_300M']

# read years from SURFRAD path
HOMEPATH = pathlib.Path('C:/Users/SFValidation3/Desktop/Mark Mikofski')
BASEDIR = HOMEPATH / 'SURFRAD'
SURFRAD_PATH = BASEDIR / 'Sioux_Falls_SD'
YEARS = list(str(y) for y in SURFRAD_PATH.iterdir())

TMY2_PATH = BASEDIR / '14944.tm2'
TMY3_PATH = BASEDIR / '726510TYA.CSV'
TMY3_PATH_2 = BASEDIR / '726515TYA.CSV'
# TMY3_PATH_3 = BASEDIR / '<tmy3filename>.CSV'
# TMY3_PATH_4 = BASEDIR / '<tmy3filename>.CSV'

DAYMINUTES = 24*60
KELVINS = 273.15


def read_surfrad_year(year_path, surfrad_path=SURFRAD_PATH):
    LOGGER.info('year: %s', year_path)
    yearpath = surfrad_path / year_path
    data = [pvlib.iotools.read_surfrad(f) for f in yearpath.iterdir()]
    dfs, heads = zip(*data)
    df = pd.concat(dfs)
    header = heads[0]
    return df, header


def estimate_air_temp(year_start, surfrad, lat, lon, cs):
    """
    Use clear sky temps scaled by daily ratio of measured to clear sky global
    insolation.

    Parameters
    ----------
    year_start : str
        SURFRAD data year
    surfrad : pandas.DateFrame
        surfrad data frame
    lat : float
        latitude in degrees north of equator [deg]
    lon : float
        longitude in degrees east of prime meridian [deg]
    cs : pandas.DataFrame
        clear sky irradiances [W/m^2]

    Returns
    -------
    est_air_temp : pandas.DataFrame
        estimated air temperature in Celsius [C]
    temp_adj : pandas.Series
        temperature adjustment [C}
    ghi_ratio : pandas.Series
        ratio of  daily SURFRAD to clearsky GHI insolation
    daily_delta_temp : numpy.array
        daily temperature range, max - min, in Kelvin [K]
    cs_temp_air : pandas.Series
        clear sky air temperatures in Celsius [C]

    """
    daze = 367 if calendar.isleap(int(year_start)) else 366
    # create a leap year of minutes for the given year at UTC
    year_minutes = pd.date_range(
        start=year_start, freq='T', periods=daze*DAYMINUTES, tz='UTC')
    # clear sky temperature
    cs_temp_air = rdtools.clearsky_temperature.get_clearsky_tamb(
        year_minutes, lat, lon)
    # organize by day
    cs_temp_daily = cs_temp_air.values.reshape((daze, DAYMINUTES)) + KELVINS
    # get daily temperature range
    daily_delta_temp = np.array([td.max()-td.min() for td in cs_temp_daily])
    daily_delta_temp = pd.Series(
        daily_delta_temp, index=cs_temp_air.resample('D').mean().index)
    # calculate ratio of daily insolation versus clearsky
    ghi_ratio = surfrad.ghi.resample('D').sum() / cs.ghi.resample('D').sum()
    ghi_ratio = ghi_ratio.rename('ghi_ratio')
    # apply ghi ratio to next day, wrap days to start at day 1
    day1 = ghi_ratio.index[0]
    ghi_ratio.index = ghi_ratio.index + to_offset('1D')
    # set day 1 estimated air temp equal to last day
    ghi_ratio[day1] = ghi_ratio.iloc[-1]
    # fix day 1 is added last, so out of order
    ghi_ratio = ghi_ratio.sort_index()
    # scale daily temperature delta by the ratio of insolation from day before
    temp_adj = (ghi_ratio - 1.0)*daily_delta_temp[ghi_ratio.index]  # use next day
    temp_adj = temp_adj.rename('temp_adj')
    # interpolate smoothly, but fill forward minutes in last day
    est_air_temp = pd.concat(
        [cs_temp_air,
         ghi_ratio.resample('1min').interpolate(),
         temp_adj.resample('1min').interpolate()], axis=1).pad()
    # Tadj = Tcs + (GHI/CS_GHI - 1) * DeltaT 
    # if GHI/CS_GHI > 1 then adjustment > DeltaT
    est_air_temp['Adjusted Temp (C)'] = (
        est_air_temp['Clear Sky Temperature (C)'] + est_air_temp.temp_adj)
    return est_air_temp, temp_adj, ghi_ratio, daily_delta_temp, cs_temp_air


if __name__ == "__main__":

    # accumulate daily energy
    EDAILY = {}

    for year_path in YEARS:
        # year_path = YEARS[-5]
        df, header = read_surfrad_year(year_path)

        # the year is at the end of the path
        # FIXME: this only works on windows paths, use pathlib Path objects instead
        year_start = year_path.rsplit('\\', 1)[1]

        # location
        LATITUDE = header['latitude']
        LONGITUDE = header['longitude']
        ELEVATION = header['elevation']

        # zero out negative (night?) GHI
        ghi = df.ghi.values
        ghi = np.where(ghi < 0, 0, ghi)

        # get solar position
        TIMES = df.index
        sp = pvlib.solarposition.get_solarposition(
                TIMES, LATITUDE, LONGITUDE)
        solar_zenith = sp.apparent_zenith.values
        solar_azimuth = sp.azimuth.values
        zenith = sp.zenith.values

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
        LOGGER.info(f'zenith MBE: {ze_mbe}%')

        # get irrad components
        irrad = pvlib.irradiance.erbs(ghi, zenith, TIMES)
        dni = irrad.dni.values
        dhi = irrad.dhi.values
        kt = irrad.kt.values  # clearness index

        # relative air mass
        AM = pvlib.atmosphere.get_relative_airmass(solar_zenith)
        # ambient pressure
        PRESS = pvlib.atmosphere.alt2pres(ELEVATION)
        # absolute (pressure adjusted ) airmass
        AMA = pvlib.atmosphere.get_absolute_airmass(AM, PRESS)
        # calculate irradiance inputs
        dni_extra = pvlib.irradiance.get_extra_radiation(TIMES).values
        # Linke turbidity
        TL = pvlib.clearsky.lookup_linke_turbidity(TIMES, LATITUDE, LONGITUDE)
        # clear sky irradiance
        CS = pvlib.clearsky.ineichen(solar_zenith, AMA, TL, ELEVATION, dni_extra)

        # estimate air temp
        est_air_temp, temp_adj, ghi_ratio, daily_delta_temp, cs_temp_air = \
            estimate_air_temp(year_start, df,LATITUDE, LONGITUDE, CS)
        temp_air = est_air_temp['Adjusted Temp (C)'].loc[TIMES].values

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
        LOGGER.info('%s annual energy: %g[kWh]', year_start, sum(Edaily) / 1000.0)
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
    LOGGER.info('PSM3: %g[kWh]', psm3eyear)
    psm3quantile = stats.percentileofscore(EYEAR, psm3eyear)
    LOGGER.info('%g quantile of SURFRAD years', psm3quantile)

    # run TMY2
    tmy2edaily = psm3tmy.run_tmy2(TMY2_PATH, LATITUDE, LONGITUDE, CECMOD_MONO)
    tmy2eyear = sum(tmy2edaily) / 1000.0
    LOGGER.info('TMY2 (%s): %g[kWh]', TMY2_PATH, tmy2eyear)
    tmy2quantile = stats.percentileofscore(EYEAR, tmy2eyear)
    LOGGER.info('%g quantile of SURFRAD years', tmy2quantile)

    # run TMY3
    tmy3edaily = psm3tmy.run_tmy3(TMY3_PATH, LATITUDE, LONGITUDE, CECMOD_MONO)
    tmy3eyear = sum(tmy3edaily) / 1000.0
    LOGGER.info('TMY3 (%s): %g[kWh]', TMY3_PATH, tmy3eyear)
    tmy3quantile = stats.percentileofscore(EYEAR, tmy3eyear)
    LOGGER.info('%g quantile of SURFRAD years', tmy3quantile)
    # run TMY3 #2 BROOKINGS
    tmy3edaily_2 = psm3tmy.run_tmy3(TMY3_PATH_2, LATITUDE, LONGITUDE, CECMOD_MONO)
    tmy3eyear_2 = sum(tmy3edaily_2) / 1000.0
    LOGGER.info('TMY3 (%s): %g[kWh]', TMY3_PATH_2, tmy3eyear_2)
    tmy3quantile_2 = stats.percentileofscore(EYEAR, tmy3eyear_2)
    LOGGER.info('%g quantile of SURFRAD years', tmy3quantile_2)
    # run TMY3 #3
    # tmy3edaily_3 = psm3tmy.run_tmy3(TMY3_PATH_3, LATITUDE, LONGITUDE, CECMOD_MONO)
    # tmy3eyear_3 = sum(tmy3edaily_3) / 1000.0
    # LOGGER.info('TMY3 (%s): %g[kWh]', TMY3_PATH_3, tmy3eyear_3)
    # tmy3quantile_3 = stats.percentileofscore(EYEAR, tmy3eyear_3)
    # LOGGER.info('%g quantile of SURFRAD years', tmy3quantile_3)
    # run TMY3 #4
    # tmy3edaily_4 = psm3tmy.run_tmy3(TMY3_PATH_4, LATITUDE, LONGITUDE, CECMOD_MONO)
    # tmy3eyear_4 = sum(tmy3edaily_4) / 1000.0
    # LOGGER.info('TMY3 (%s): %g[kWh]', TMY3_PATH_4, tmy3eyear_4)
    # tmy3quantile_4 = stats.percentileofscore(EYEAR, tmy3eyear_4)
    # LOGGER.info('%g quantile of SURFRAD years', tmy3quantile_4)

    # stop logging
    LOGGER.setLevel(logging.CRITICAL)

    # site name:
    SITE = SURFRAD_PATH.parts[-1]

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
    # ax[1].plot([tmy3eyear_3]*2, ylim, 'k--')
    # ax[1].annotate(f' {tmy3quantile_3:g}%', (tmy3eyear_3, 0.65*ylim[1]))
    # ax[1].plot([tmy3eyear_4]*2, ylim, 'k--')
    # ax[1].annotate(f' {tmy3quantile_4:g}%', (tmy3eyear_4, 0.55*ylim[1]))
    ax[1].legend(['KDE', 'P50', 'P90', 'PSM3', 'TMY2', 'TMY3'])
    ax[1].set_title(
        f'{SITE} Distribution: P50 = {P50:g}[kWh], P90 = {P90:g}[kWh], PSM3 = {psm3eyear:g}[kWh]'
        f'\nTMY2 = {tmy2eyear:g}[kWh], TMY3(726510) = {tmy3eyear:g}[kWh], TMY3(726515) = {tmy3eyear_2:g}[kWh]'
        # f'\nTMY3(722356) = {tmy3eyear_2:g}[kWh], TMY3(723340) = {tmy3eyear_3:g}[kWh], TMY3(723306) = {tmy3eyear_4:g}[kWh]'
    )
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
        # 'tmy3eyear_3': tmy3eyear_3, 'tmy3quantile_3': tmy3quantile_3,
        # 'tmy3eyear_4': tmy3eyear_4, 'tmy3quantile_4': tmy3quantile_4,
    }
    with open(f'{SITE}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
