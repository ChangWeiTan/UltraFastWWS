package distances.classic;

import distances.ElasticDistances;
import results.WarpingPathResults;

public class ERP extends ElasticDistances {
    public double distance(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);
        double[] prev = new double[minLen];
        double[] curr = new double[minLen];

        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }

        for (i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            tmp = prev;
            prev = curr;
            curr = tmp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > (m - 1)) {
                right = (m - 1);
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    d1 = sqDist(first[i], g);
                    d2 = sqDist(second[j], g);
                    d12 = sqDist(first[i], second[j]);

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) &&
                                (((prev[j - 1] + d12) > (curr[j - 1] + d2)) &&
                                        ((curr[j - 1] + d2) <= (prev[j] + d1))))) {
                            cost = curr[j - 1] + d2;
                        } else if (j == 0 || prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2) {
                            cost = prev[j] + d1;
                        } else {
                            cost = prev[j - 1] + d12;
                        }
                    } else {
                        cost = d12;
                    }

                    curr[j] = cost;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return (curr[m - 1]);
    }

    public WarpingPathResults distanceExt(double[] first, double[] second, final double g, final double bandSize) {
        final int m = first.length;
        final int n = second.length;
        final int minLen = Math.min(m, n);
        final int band = getWindowSize(minLen, bandSize);

        final int[][] minWarpingWindow = new int[minLen][minLen];
        double[] prev = new double[minLen];
        double[] curr = new double[minLen];

        double diff, d1, d2, d12, cost;
        int i, j, left, right, absIJ;

        double[] tmp;
        if (n < m) {
            tmp = first;
            first = second;
            second = tmp;
        }


        minWarpingWindow[0][0] = 0;
        for (i = 0; i < m; i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            double[] temp = prev;
            prev = curr;
            curr = temp;

            left = i - (band + 1);
            if (left < 0) {
                left = 0;
            }
            right = i + (band + 1);
            if (right > (m - 1)) {
                right = (m - 1);
            }

            for (j = left; j <= right; j++) {
                absIJ = Math.abs(i - j);
                if (absIJ <= band) {
                    d1 = sqDist(first[i], g);
                    d2 = sqDist(second[j], g);
                    d12 = sqDist(first[i], second[j]);

                    if ((i + j) != 0) {
                        if ((i == 0) || ((j != 0) &&
                                (((prev[j - 1] + d12) >= (curr[j - 1] + d2)) &&
                                        ((curr[j - 1] + d2) <= (prev[j] + d1))))) {
                            // del
                            cost = curr[j - 1] + d2;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i][j - 1]);
                        } else if (j == 0 || prev[j - 1] + d12 >= prev[j] + d1 && prev[j] + d1 <= curr[j - 1] + d2) {
                            // ins
                            cost = prev[j] + d1;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j]);
                        } else {
                            // match
                            cost = prev[j - 1] + d12;
                            minWarpingWindow[i][j] = Math.max(absIJ, minWarpingWindow[i - 1][j - 1]);
                        }
                    } else {
                        cost = 0;
                        minWarpingWindow[i][j] = 0;
                    }

                    curr[j] = cost;
                    // steps[i][j] = step;
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
        }

        return new WarpingPathResults(curr[m - 1], minWarpingWindow[n - 1][m - 1]);
    }

    public static int getWindowSize(int n, double bandSize) {
        return (int) Math.ceil(bandSize * n);
    }
}
