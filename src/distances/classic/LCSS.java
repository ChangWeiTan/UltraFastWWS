package distances.classic;

import distances.ElasticDistances;
import results.WarpingPathResults;

public class LCSS extends ElasticDistances {
    public double distance(final double[] first, final double[] second, final double epsilon, final int delta) {
        final int m = first.length;
        final int n = second.length;
        int[][] lcss = new int[m + 1][n + 1];
        int i, j;

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (Math.abs(first[i]-second[j]) < epsilon) {
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                } else if (delta == 0) {
                    lcss[i + 1][j + 1] = lcss[i][j];
                } else
                    lcss[i + 1][j + 1] = Math.max(lcss[i][j + 1], lcss[i + 1][j]);
            }
        }

        int max = -1;
        for (i = 1; i < m + 1; i++) {
            if (lcss[m][i] > max) {
                max = lcss[m][i];
            }
        }
        return 1.0 - 1.0 * lcss[m][n] / m;
    }

    public WarpingPathResults distanceExt(final double[] first, final double[] second, final double epsilon, final int delta) {
        final int m = first.length;
        final int n = second.length;
        int i, j, absIJ;
        final int[][] lcss = new int[m + 1][n + 1];
        final int[][] minDelta = new int[m + 1][n + 1];

        for (i = 0; i < m; i++) {
            for (j = i - delta; j <= i + delta; j++) {
                if (j < 0) {
                    j = -1;
                } else if (j >= n) {
                    j = i + delta;
                } else if (Math.abs(first[i]-second[j]) < epsilon) {
                    absIJ = Math.abs(i - j);
                    lcss[i + 1][j + 1] = lcss[i][j] + 1;
                    minDelta[i + 1][j + 1] = Math.max(absIJ, minDelta[i][j]);
                } else if (delta == 0) {
                    lcss[i + 1][j + 1] = lcss[i][j];
                    minDelta[i + 1][j + 1] = 0;
                } else if (lcss[i][j + 1] > lcss[i + 1][j]) {
                    lcss[i + 1][j + 1] = lcss[i][j + 1];
                    minDelta[i + 1][j + 1] = minDelta[i][j + 1];
                } else {
                    lcss[i + 1][j + 1] = lcss[i + 1][j];
                    minDelta[i + 1][j + 1] = minDelta[i + 1][j];
                }
            }
        }

        int max = -1, maxR = -1;
        for (i = 1; i < m + 1; i++) {
            if (lcss[m][i] > max) {
                max = lcss[m][i];
                maxR = minDelta[m][i];
            }
        }
        return new WarpingPathResults(1.0 - 1.0 * lcss[m][n] / m, maxR);
    }
}
