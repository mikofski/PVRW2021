import os
import warnings
import pandas as pd
import pvlib


NREL_API_KEY = os.getenv('NREL_API_KEY', 'DEMO_KEY')
EMAIL = os.getenv('EMAIL', 'bwana.marko@yahoo.com')


def run_psm3tmy(latitude, longitude, cecmod, apikey=NREL_API_KEY, email=EMAIL):
    """
    Run simulation with PSM3 TMY

    Parameters
    ----------
    latitude : 
    """

    header, data = pvlib.iotools.get_psm3(latitude, longitude, apikey, email)
    times = pd.DatetimeIndex(
        dt.replace(year=1990) for dt in data.index.to_pydatetime())


    # get solar position
    data.index = pd.DatetimeIndex(times)
    sp = pvlib.solarposition.get_solarposition(
            times, latitude, longitude)
    solar_zenith = sp.apparent_zenith.values
    solar_azimuth = sp.azimuth.values
    dni = data.DNI.values
    ghi = data.GHI.values
    dhi = data.DHI.values
    surface_albedo = data['Surface Albedo'].values
    temp_air = data.Temperature.values
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values

    # trackers
    tracker = pvlib.tracking.singleaxis(solar_zenith, solar_azimuth)
    surface_tilt = tracker['surface_tilt']
    surface_azimuth = tracker['surface_azimuth']
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, dni_extra=dni_extra, model='haydavies')
    aoi = tracker['aoi']

    # irrad in plane of array
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
            surface_tilt, ghi, albedo=surface_albedo)
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
                cecmod.alpha_sc, cecmod.a_ref,
                cecmod.I_L_ref, cecmod.I_o_ref,
                cecmod.R_sh_ref, cecmod.R_s, cecmod.Adjust)
        mpp = pvlib.pvsystem.max_power_point(*cecparams, method='newton')
    mpp = pd.DataFrame(mpp, index=times)
    return mpp.p_mp.resample('D').sum()


def run_tmy2(tmy2path, latitude, longitude, cecmod):
    """Run simulation with PSM3 TMY"""

    data, header = pvlib.iotools.read_tmy2(tmy2path)
    times = pd.DatetimeIndex(
        dt.replace(year=1990) for dt in data.index.to_pydatetime())

    # get solar position
    data.index = times
    sp = pvlib.solarposition.get_solarposition(
            times, latitude, longitude)
    solar_zenith = sp.apparent_zenith.values
    solar_azimuth = sp.azimuth.values
    dni = data.DNI.values
    ghi = data.GHI.values
    dhi = data.DHI.values
    # TMY2 has no albedo
    temp_air = data.DryBulb.values / 10.0
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values

    # trackers
    tracker = pvlib.tracking.singleaxis(solar_zenith, solar_azimuth)
    surface_tilt = tracker['surface_tilt']
    surface_azimuth = tracker['surface_azimuth']
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, dni_extra=dni_extra, model='haydavies')
    aoi = tracker['aoi']

    # irrad in plane of array
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
                cecmod.alpha_sc, cecmod.a_ref,
                cecmod.I_L_ref, cecmod.I_o_ref,
                cecmod.R_sh_ref, cecmod.R_s, cecmod.Adjust)
        mpp = pvlib.pvsystem.max_power_point(*cecparams, method='newton')
    mpp = pd.DataFrame(mpp, index=times)
    return mpp.p_mp.resample('D').sum()


def run_tmy3(tmy3path, latitude, longitude, cecmod):
    """Run simulation with PSM3 TMY"""

    data, header = pvlib.iotools.read_tmy3(tmy3path)
    times = pd.DatetimeIndex(
        dt.replace(year=1990) for dt in data.index.to_pydatetime())

    # get solar position
    data.index = times
    sp = pvlib.solarposition.get_solarposition(
            times, latitude, longitude)
    solar_zenith = sp.apparent_zenith.values
    solar_azimuth = sp.azimuth.values
    dni = data.DNI.values
    ghi = data.GHI.values
    dhi = data.DHI.values
    albedo = data.Alb.values
    temp_air = data.DryBulb.values
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values

    # trackers
    tracker = pvlib.tracking.singleaxis(solar_zenith, solar_azimuth)
    surface_tilt = tracker['surface_tilt']
    surface_azimuth = tracker['surface_azimuth']
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            dni, ghi, dhi, dni_extra=dni_extra, model='haydavies')
    aoi = tracker['aoi']

    # irrad in plane of array
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
            surface_tilt, ghi, albedo=albedo)
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
                cecmod.alpha_sc, cecmod.a_ref,
                cecmod.I_L_ref, cecmod.I_o_ref,
                cecmod.R_sh_ref, cecmod.R_s, cecmod.Adjust)
        mpp = pvlib.pvsystem.max_power_point(*cecparams, method='newton')
    mpp = pd.DataFrame(mpp, index=times)
    return mpp.p_mp.resample('D').sum()
