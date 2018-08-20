from astropy.table import Table
from astroquery.gaia import Gaia


soubiran_2013 = Table.read("data/Soubiran_2013.fits")
N = len(soubiran_2013)

# Get gaia source IDs and we can do the cross-match from there.
source_ids = []
for i, hip in enumerate(soubiran_2013["HIP"]):

    job = Gaia.launch_job("""
        SELECT gaia.source_id
        FROM gaiadr2.gaia_source AS gaia,
             gaiadr2.hipparcos2_best_neighbour AS hip
        WHERE hip.original_ext_source_id = '{:.0f}'
        AND   hip.source_id = gaia.source_id
        """.format(hip))

    results = job.get_results()
    if len(results) == 0:
        source_id = -1
    else:
        source_id = results["source_id"][0]

    print("{}/{}: HIP {} = Gaia DR2 {}".format(i, N, hip, source_id))
    source_ids.append(source_id)

import numpy as np
soubiran_2013["source_id"] = np.array(source_ids, dtype=np.int64)

OK = np.isfinite(soubiran_2013["source_id"])
soubiran_2013 = soubiran_2013[OK]

soubiran_2013.write("data/Soubiran_2013-xm-Gaia.fits", overwrite=True)
