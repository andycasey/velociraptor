
"""
Create a mask that will be useful for science verification.
"""

import numpy as np
import pickle
from astropy.io import fits
from astropy.table import Table



data = fits.open("data/gaia-sources-for-npm-colsubset.fits")[1].data

cross_match_paths = [
    "data/soubiran-2013-xm-gaia.fits",
    "data/sb9_xm_gaia.fits",
    "data/apw-highK-unimodal-xm-gaia.fits",
    "data/apw-lnK-percentiles-xm-gaia.fits",
    "data/huang-apogee-rv-standards-xm-gaia.fits"
]


def cross_match(A_source_ids, B_source_ids):

    A = np.array(A_source_ids, dtype=np.long)
    B = np.array(B_source_ids, dtype=np.long)

    ai = np.where(np.in1d(A, B))[0]
    bi = np.where(np.in1d(B, A))[0]
    
    assert len(ai) == len(bi)
    ai = ai[np.argsort(A[ai])]
    bi = bi[np.argsort(B[bi])]

    assert all(A[ai] == B[bi])
    return (ai, bi)



mask = np.zeros(len(data), dtype=bool)

for path in cross_match_paths:

    t = Table.read(path)
    t = t.group_by("source_id")
    t = t[t.groups.indices[:-1]]

    vl_ids, t_ids = cross_match(data["source_id"], t["source_id"])

    mask[vl_ids] = True

    print(f"Cross-matching with {path} revealed {len(vl_ids)} sources")


with open("sv.mask", "wb") as fp:
    pickle.dump(mask, fp)

print(f"There are {sum(mask)} sources for science verification in sv.mask")