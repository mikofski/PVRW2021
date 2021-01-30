#! python

import ftplib
import io
import json
import logging
import pathlib
import sys
import threading
import time
import queue

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

NOAA = r'aftp.cmdl.noaa.gov'
SURFRAD = r'data/radiation/surfrad'
SITES = [
    'Bondville_IL',     # 1995 - 2021
    'Boulder_CO',       # 1995 - 2021
    'Desert_Rock_NV'    # 1998 - 2021
    'Fort_Peck_MT',     # 1995 - 2021
    'Goodwin_Creek_MS', # 1995 - 2021
    'Penn_State_PA',    # 1998 - 2021
    'Sioux_Falls_SD',   # 2003 - 2021
]
Q = queue.Queue()


def save_surfrad_year(site, year, noaa=NOAA, surfrad=SURFRAD, q=Q):
    """
    A target for threading that get's surfrad data from FTP.
    """
    LOGGER.debug('open FTP connection to %s', noaa)
    ftp = ftplib.FTP(noaa)
    LOGGER.debug('login')
    ftp.login()
    LOGGER.debug('change cwd to %s/%s,%s', surfrad, site, year)
    ftp.cwd(f'{surfrad}/{site}/{year}')
    siteyear_path = pathlib.Path(site) / year
    LOGGER.debug('local path: %s', siteyear_path)
    days = []
    ftp.retrlines('NLST', lambda _: days.append(_))
    ndays = len(days)
    LOGGER.debug('number of days in %s:%s = %d', site, year, ndays)
    for day in days:
        p = siteyear_path / day
        if p.exists():
            LOGGER.debug('file already exists: %s', str(p))
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f'downloading: {day}')
        with p.open('wb') as fp:
            ftp.retrbinary(f'RETR {day}', fp.write)
    LOGGER.debug('download of %s:%s complete!', site, year)
    ftp.quit()
    ftp.close()
    q.put({f'{year}': ndays})


if __name__ == '__main__':

    if len(sys.argv) < 2:
        LOGGER.error('what site?')
        sys.exit(-1)
    site = sys.argv[1]

    # get years
    ftp = ftplib.FTP(NOAA)
    ftp.login()
    ftp.cwd(f'{SURFRAD}/{site}')
    years = []
    ftp.retrlines('NLST', lambda _: years.append(_))
    ftp.quit()
    ftp.close()

    maxconn = 5
    siteyear_meta = {}
    threads = []
    for year in years:
        if year == "README":
            continue
        LOGGER.debug(f'year: {year}')
        t = threading.Thread(target=save_surfrad_year, args=(site, year), name=year)
        threads.append(t)
        t.start()
        LOGGER.debug('thread: %s', t)
        time.sleep(5)
        if len(threads) % maxconn == 0:
            # Q might be empty if the first years are still downloading
            threads[0].join()  # join the first thread in the list
            deadthread = threads.pop(0)
            LOGGER.debug('thread %s complete!', deadthread)
            siteyear_ndays = Q.get_nowait()
            siteyear_meta.update(siteyear_ndays)
            LOGGER.debug('queue: %s', siteyear_ndays)
    # get last records less than maxconn
    for t in threads:
        t.join()
        LOGGER.debug('thread %s complete!', t)
        siteyear_ndays = Q.get_nowait()
        siteyear_meta.update(siteyear_ndays)
        LOGGER.debug('queue: %s', siteyear_ndays)
    print(json.dumps(siteyear_meta, indent=2))
    with open(f'{site}.meta', 'w') as f:
        json.dump(siteyear_meta, f, indent=2)
