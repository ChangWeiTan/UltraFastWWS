package distances.classic;

import distances.ElasticDistances;

public class MSM extends ElasticDistances {
    public double distance(final double[] first, final double[] second, final double c) {
        final int m = first.length;
        final int n = second.length;
        final double[][] matrixD = new double[m][n];
        int i, j;
        double d1, d2, d3;

        // Initialization
        matrixD[0][0] = Math.abs(first[0] - second[0]);
        for (i = 1; i < m; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + editCost(first[i], first[i - 1], second[0], c);
        }
        for (i = 1; i < n; i++) {
            matrixD[0][i] = matrixD[0][i - 1] + editCost(second[i], first[0], second[i - 1], c);
        }

        // Main Loop
        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                d1 = matrixD[i - 1][j - 1] + Math.abs(first[i] - second[j]);
                d2 = matrixD[i - 1][j] + editCost(first[i], first[i - 1], second[j], c);
                d3 = matrixD[i][j - 1] + editCost(second[j], first[i], second[j - 1], c);
                matrixD[i][j] = Math.min(d1, Math.min(d2, d3));
            }
        }
        // Output
        return matrixD[m - 1][n - 1];
    }

    public double distance(final double[] first, final double[] second, final double c, final double cutOffValue) {
        final int m = first.length;
        final int n = second.length;
        final double[][] matrixD = new double[m][n];
        int i, j;
        double d1, d2, d3;
        double min;

        // Initialization
        matrixD[0][0] = Math.abs(first[0] - second[0]);
        for (i = 1; i < m; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + editCost(first[i], first[i - 1], second[0], c);
        }
        for (i = 1; i < n; i++) {
            matrixD[0][i] = matrixD[0][i - 1] + editCost(second[i], first[0], second[i - 1], c);
        }

        // Main Loop
        for (i = 1; i < m; i++) {
            min = cutOffValue;
            for (j = 1; j < n; j++) {
                d1 = matrixD[i - 1][j - 1] + Math.abs(first[i] - second[j]);
                d2 = matrixD[i - 1][j] + editCost(first[i], first[i - 1], second[j], c);
                d3 = matrixD[i][j - 1] + editCost(second[j], first[i], second[j - 1], c);
                matrixD[i][j] = Math.min(d1, Math.min(d2, d3));

                if (matrixD[i][j] >= cutOffValue) {
                    matrixD[i][j] = Double.MAX_VALUE;
                }

                if (matrixD[i][j] < min) {
                    min = matrixD[i][j];
                }
            }
            if (min >= cutOffValue) {
                return Double.MAX_VALUE;
            }
        }
        // Output
        return matrixD[m - 1][n - 1];
    }

    private double editCost(final double new_point, final double x, final double y, final double c) {
        double dist;

        if (((x <= new_point) && (new_point <= y)) || ((y <= new_point) && (new_point <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
        }

        return dist;
    }
}
