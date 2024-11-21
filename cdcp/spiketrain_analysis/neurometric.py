import numpy as np
import warnings
from cdcp.behavior.psychometric import FourParameterLogistic, fit_FourParameterLogistic


def get_interp_points_dists_from_similarity_matrix(
    interp_points_this_unit, similarity_matrix, n_interp_point_bins=128
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # with np.errstate(invalid="ignore"):
        interp_points = []
        dists = []
        for ri, interp_point in enumerate(interp_points_this_unit):
            mask = interp_points_this_unit > (n_interp_point_bins / 2) - 1

            # skip if there isn't anything to compare
            if np.sum(mask) < 1:
                continue
            if np.sum(mask == False) < 1:
                continue

            interp_points.append(interp_point)

            a = np.nanmean(similarity_matrix[ri][mask])
            b = np.nanmean(similarity_matrix[ri][mask == False])
            dist = a / (a + b)

            dists.append(dist)
        return interp_points, dists
