"""Kozeny-Carmen fitting for PGE 323M Assignment 6.

Copyright 2018-2026 John T. Foster. Licensed under Apache-2.0.
Migrated from the original assignment notebook to a Python module.
"""

import numpy as np
import pandas as pd


class KozenyCarmen:
    """Read porosity/permeability observations and fit the K-C model."""

    def __init__(self, filename):
        """Read filename with pandas and add the KC model column."""
        raise NotImplementedError("Complete __init__")

    def kc_model(self):
        """Add the vectorized phi**3 / (1 - phi)**2 column; return None."""
        raise NotImplementedError("Complete kc_model")

    def least_squares(self, A, b):
        """Solve the normal equations for full-column-rank A and 1-D b."""
        raise NotImplementedError("Complete least_squares")

    def fit(self):
        """Return the two coefficients in (intercept, slope) order."""
        raise NotImplementedError("Complete fit")

    def fit_through_zero(self):
        """Fit mirrored observations with an intercept column; return slope."""
        raise NotImplementedError("Complete fit_through_zero")
