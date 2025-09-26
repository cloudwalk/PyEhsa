import numpy as np
import pandas as pd
from scipy import stats


class MannKendall:
    @staticmethod
    def mann_kendall_test(x):
        """
        Perform Mann-Kendall test with improved handling of edge cases
        """
        n = len(x)

        # Handle insufficient data
        if n <= 1:
            return {"x": None, "tau": 0, "sl": 1.0, "S": 0, "D": 0, "varS": 0}

        # Convert input to numpy array and remove NaN values
        x = np.array(x)
        x = x[~np.isnan(x)]
        n = len(x)  # Update n after filtering NaN values

        if n <= 1:
            return {"x": None, "tau": 0, "sl": 1.0, "S": 0, "D": 0, "varS": 0}

        # Calculate S statistic
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = x[j] - x[i]
                if abs(diff) > 1e-10:  # Use small epsilon instead of exact zero
                    S += np.sign(diff)

        # Calculate denominator D
        D = n * (n - 1) / 2

        # Handle case where D is zero
        if D < 1e-10:
            return {"x": None, "tau": 0, "sl": 1.0, "S": int(S), "D": int(D), "varS": 0}

        # Calculate tau
        tau = S / D

        # Calculate variance
        varS = (n * (n - 1) * (2 * n + 5)) / 18

        # Adjust variance for ties
        unique_x = np.unique(x)
        if len(unique_x) < len(x):
            t = np.zeros(len(unique_x))
            for i, val in enumerate(unique_x):
                t[i] = np.sum(
                    np.abs(x - val) < 1e-10
                )  # Use small epsilon for comparison
            v_adjust = np.sum(t * (t - 1) * (2 * t + 5)) / 18
            varS -= v_adjust

        # Calculate p-value
        if varS <= 1e-10:
            sl = 1.0
        else:
            if abs(S) <= 1e-10:
                sl = 1.0
            else:
                # Calculate z-score with continuity correction
                if S > 0:
                    z = (S - 1) / np.sqrt(varS)
                else:
                    z = (S + 1) / np.sqrt(varS)
                # Two-tailed p-value
                sl = 2 * (1 - stats.norm.cdf(abs(z)))

        # Format results
        tau = float(format(tau, ".15f"))
        sl = float(format(sl, ".15f")) if not np.isnan(sl) else 1.0
        varS = float(format(varS, ".15f"))

        return {"x": None, "tau": tau, "sl": sl, "S": int(S), "D": int(D), "varS": varS}
