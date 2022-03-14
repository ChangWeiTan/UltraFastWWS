package distances.classic;

import distances.ElasticDistances;
import results.WarpingPathResults;
import utils.GenericTools;

public class ERP extends ElasticDistances {
    public double distance(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);
        double[] prev = new double[minLen + 1];
        double[] curr = new double[minLen + 1];

        double d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

        for (i = 0; i <= m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            tmp = prev;
            prev = curr;
            curr = tmp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > m) {
                right = m;
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    if ((i + j) == 0)
                        cost = 0;
                    else if (i == 0)
                        cost = curr[j - 1] + sqDist(second[j - 1], g);
                    else if (j == 0)
                        cost = prev[j] + sqDist(first[i - 1], g);
                    else {
                        d1 = sqDist(first[i - 1], g);
                        d2 = sqDist(second[j - 1], g);
                        d12 = sqDist(first[i - 1], second[j - 1]);

                        if (((prev[j - 1] + d12) > (curr[j - 1] + d2)) && ((curr[j - 1] + d2) <= (prev[j] + d1))) {
                            cost = curr[j - 1] + d2;
                        } else if ((prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2)) {
                            cost = prev[j] + d1;
                        } else {
                            cost = prev[j - 1] + d12;
                        }
                    }

                    curr[j] = cost;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return (curr[m]);
    }

    public WarpingPathResults distanceExt(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);
        final int[][] minWarpingWindow = new int[minLen + 1][minLen + 1];
        double[] prev = new double[minLen + 1];
        double[] curr = new double[minLen + 1];

        double d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

        minWarpingWindow[0][0] = 0;
        for (i = 0; i <= m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            tmp = prev;
            prev = curr;
            curr = tmp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > m) {
                right = m;
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    if ((i + j) == 0) {
                        cost = 0;
                        minWarpingWindow[i][j] = 0;
                    } else if (i == 0) {
                        cost = curr[j - 1] + sqDist(second[j - 1], g);
                        minWarpingWindow[i][j] = j;
                    } else if (j == 0) {
                        cost = prev[j] + sqDist(first[i - 1], g);
                        minWarpingWindow[i][j] = i;
                    } else {
                        d1 = sqDist(first[i - 1], g);
                        d2 = sqDist(second[j - 1], g);
                        d12 = sqDist(first[i - 1], second[j - 1]);

                        if (((prev[j - 1] + d12) > (curr[j - 1] + d2)) && ((curr[j - 1] + d2) <= (prev[j] + d1))) {
                            cost = curr[j - 1] + d2;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j - 1]);
                        } else if ((prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2)) {
                            cost = prev[j] + d1;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j]);
                        } else {
                            cost = prev[j - 1] + d12;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j - 1]);
                        }
                    }

                    curr[j] = cost;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return new WarpingPathResults(curr[m], minWarpingWindow[n][m]);
    }

    public static int getWindowSize(int n, double bandSize) {
        return (int) Math.ceil(bandSize * n);
    }
}
