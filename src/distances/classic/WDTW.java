package distances.classic;

import distances.ElasticDistances;
import results.WarpingPathResults;
import utils.GenericTools;

public class WDTW extends ElasticDistances {
    public double distance(double[] first, double[] second, double[] weightVector) {
        int m = first.length;
        int n = second.length;
        final double[][] matrixD = new double[n][m];
        double diff;
        double minDistance;

        //first value
        diff = first[0] - second[0];
        matrixD[0][0] = weightVector[0] * diff * diff;

        //first column
        for (int i = 1; i < m; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (int j = 1; j < n; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                //calculate classifiers.nearestNeighbour.distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first[i] - second[j];
                matrixD[i][j] = minDistance + weightVector[Math.abs(i - j)] * diff * diff;
            }
        }
        return matrixD[m - 1][n - 1];
    }

    public double distance(double[] first, double[] second, double[] weightVector, double cutoff) {
        boolean tooBig;
        int m = first.length;
        int n = second.length;
        final double[][] matrixD = new double[n][m];
        double diff;
        double minDistance;

        //first value
        diff = first[0] - second[0];
        matrixD[0][0] = weightVector[0] * diff * diff;
        if (matrixD[0][0] > cutoff) {
            return Double.POSITIVE_INFINITY;
        }

        //first column
        for (int i = 1; i < m; i++) {
            diff = first[i] - second[0];
            matrixD[i][0] = matrixD[i - 1][0] + weightVector[i] * diff * diff;
        }

        //top row
        for (int j = 1; j < n; j++) {
            diff = first[0] - second[j];
            matrixD[0][j] = matrixD[0][j - 1] + weightVector[j] * diff * diff;
        }

        //warp rest
        for (int i = 1; i < m; i++) {
            tooBig = !(matrixD[i][0] < cutoff);
            for (int j = 1; j < n; j++) {
                //calculate classifiers.nearestNeighbour.distances
                minDistance = Math.min(matrixD[i][j - 1], Math.min(matrixD[i - 1][j], matrixD[i - 1][j - 1]));
                diff = first[i] - second[j];
                matrixD[i][j] = minDistance + weightVector[Math.abs(i - j)] * diff * diff;
                if (tooBig && matrixD[i][j] < cutoff) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return matrixD[m - 1][n - 1];
    }
}
