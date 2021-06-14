# add TMY to GEOJSON

import copy
import json
import pvlib
import pandas as pd
import numpy as np

with open ('../tmy_sites.json') as f:
    tmy_sites = json.load(f)

geojson_template = {
    'geometry': {
        'type': 'Point',
        'coordinates': [-105.25, 40.016666666666666]},
    'type': 'Feature',
    'properties': {
        'station': 'Boulder, CO (TMY2)',
        'elevation (m)': 1634}}

geojson_output = []

for k, v in tmy_sites.items():
    print(f'SURFRAD site: {k}')
    for sitename, filename in v.items():
        print(f'TMY site: {sitename}')
        site_geojson = copy.deepcopy(geojson_template)
        if filename.endswith('tm2'):
            _, filemeta = pvlib.iotools.read_tmy2(filename)
            site_geojson['geometry']['coordinates'] = [filemeta['longitude'], filemeta['latitude']]
            site_geojson['properties']['station'] = f"{filemeta['City']}, {filemeta['State']} (TMY2)"
            site_geojson['properties']['elevation (m)'] = filemeta['altitude']
            site_geojson['properties']['WBAN'] = filemeta['WBAN']
        else:
            _, filemeta = pvlib.iotools.read_tmy3(filename)
            site_geojson['geometry']['coordinates'] = [filemeta['longitude'], filemeta['latitude']]
            site_geojson['properties']['station'] = f"{filemeta['Name'][1:-1]} (TMY3)"
            site_geojson['properties']['elevation (m)'] = filemeta['altitude']
            site_geojson['properties']['USAF'] = filemeta['USAF']
        geojson_output.append(site_geojson)
        print(f'TMY file: {filename}')

with open('tmy_sites_geo.json', 'w') as g:
    json.dump(geojson_output, g, indent=2)

sites_df = {}

for site in geojson_output:
    sites_df[site['properties']['station']] = {
        'latitude': site['geometry']['coordinates'][1],
        'longitude': site['geometry']['coordinates'][0],
        'elevation': site['properties']['elevation (m)'],
        'WBAN': str(site['properties'].get('WBAN')),
        'USAF': str(site['properties'].get('USAF'))}

df_sites = pd.DataFrame(sites_df)

df_sites.T.to_csv('TMY_sites_df.csv')

with open('TMY_sites_df.json','w') as h:
    json.dump(sites_df, h, indent=2)
