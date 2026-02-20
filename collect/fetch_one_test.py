from fetcher import NRAOQuery, COLS, TIMESPANS, INSTRUMENTS, DATAPRODS, CONFIGS, BANDS, PROPRIETARY
import astropy.units as u
from astropy.coordinates import SkyCoord

# 0005+383
pos = SkyCoord("00h05m57.175409s","38d20'15.148570\"", frame="icrs")

q = (
    NRAOQuery(limit=3)
    .where_timespan(TIMESPANS.FROM_2016_SEP)
    .where_in_circle(pos, 20 * u.arcsec)
    .where_instruments(INSTRUMENTS.VLA_VARIANTS())
    .where_dataproduct(DATAPRODS.VISIBILITY)
    .where_configs([CONFIGS.A, CONFIGS.B, CONFIGS.C, CONFIGS.D])
    .where_band(BANDS.C)
    .where_proprietary_status(PROPRIETARY.PUBLIC)
    .order_by("t_min DESC")
)

df = q.get()
print(df.columns)
print(df.head())
url = df.loc[0, "access_url"]

from product_details import fetch_product_details, summarize_product_details, print_summary

details = fetch_product_details(df.loc[0, "access_url"])
summary = summarize_product_details(details, cal_center=pos)
print_summary(summary)


print(details.keys())


print('done')
