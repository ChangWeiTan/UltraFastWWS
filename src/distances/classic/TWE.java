package distances.classic;

import distances.ElasticDistances;
import utils.GenericTools;

public class TWE extends ElasticDistances {
//    public double distance(double[] first, double[] second, double nu, double lambda) {
//        final int m = first.length;
//        final int n = second.length;
//        final double[][] D = new double[m + 1][m + 1];
//        final double[] Di1 = new double[m + 1];
//        final double[] Dj1 = new double[n + 1];
//
//        double diff, dist;
//        double dmin, htrans;
//        int i, j;
//
//        // local costs initializations
//        for (j = 1; j <= n; j++) {
//            if (j > 1) {
//                Dj1[j] = second[j - 2] - second[j - 1];
//                Dj1[j] = Dj1[j] * Dj1[j];
//            } else {
//                Dj1[j] = second[j - 1] * second[j - 1];
//            }
//        }
//
//        for (i = 1; i <= m; i++) {
//            if (i > 1) {
//                Di1[i] = first[i - 2] - first[i - 1];
//                Di1[i] = Di1[i] * Di1[i];
//            } else {
//                Di1[i] = first[i - 1] * first[i - 1];
//            }
//
//            for (j = 1; j <= n; j++) {
//                D[i][j] = first[i - 1] - second[j - 1];
//                D[i][j] = D[i][j] * D[i][j];
//                if (i > 1 && j > 1) {
//                    diff = first[i - 2] - second[j - 2];
//                    D[i][j] += diff * diff;
//                }
//            }
//        }
//
//        // border of the cost matrix initialization
//        D[0][0] = 0;
//        for (i = 1; i <= m; i++) {
//            D[i][0] = D[i - 1][0] + Di1[i];
//        }
//        for (j = 1; j <= n; j++) {
//            D[0][j] = D[0][j - 1] + Dj1[j];
//        }
//
//        for (i = 1; i <= m; i++) {
//            for (j = 1; j <= n; j++) {
//                htrans = Math.abs(i - j);
//                if (j > 1 && i > 1) {
//                    htrans *= 2;
//                }
//                dmin = D[i - 1][j - 1] + nu * htrans + D[i][j];
//
//                dist = Di1[i] + D[i - 1][j] + lambda + nu;
//                if (dmin > dist) {
//                    dmin = dist;
//                }
//                dist = Dj1[j] + D[i][j - 1] + lambda + nu;
//                if (dmin > dist) {
//                    dmin = dist;
//                }
//
//                D[i][j] = dmin;
//            }
//        }
//
//        dist = D[m][n];
//        return dist;
//    }
//
//    public double distance(double[] first, double[] second, double nu, double lambda, double cutoff) {
//        boolean tooBig;
//        final int m = first.length;
//        final int n = second.length;
//        final double[][] D = new double[m + 1][m + 1];
//        final double[] Di1 = new double[m + 1];
//        final double[] Dj1 = new double[n + 1];
//
//        double diff, dist;
//        double dmin, htrans;
//        int i, j;
//
//        // local costs initializations
//        for (j = 1; j <= n; j++) {
//            if (j > 1) {
//                Dj1[j] = second[j - 2] - second[j - 1];
//                Dj1[j] = Dj1[j] * Dj1[j];
//            } else {
//                Dj1[j] = second[j - 1] * second[j - 1];
//            }
//        }
//
//        for (i = 1; i <= m; i++) {
//            if (i > 1) {
//                Di1[i] = first[i - 2] - first[i - 1];
//                Di1[i] = Di1[i] * Di1[i];
//            } else {
//                Di1[i] = first[i - 1] * first[i - 1];
//            }
//
//            for (j = 1; j <= n; j++) {
//                D[i][j] = first[i - 1] - second[j - 1];
//                D[i][j] = D[i][j] * D[i][j];
//                if (i > 1 && j > 1) {
//                    diff = first[i - 2] - second[j - 2];
//                    D[i][j] += diff * diff;
//                }
//            }
//        }
//
//        // border of the cost matrix initialization
//        D[0][0] = 0;
//        for (i = 1; i <= m; i++) {
//            D[i][0] = D[i - 1][0] + Di1[i];
//        }
//        for (j = 1; j <= n; j++) {
//            D[0][j] = D[0][j - 1] + Dj1[j];
//        }
//
//        for (i = 1; i <= m; i++) {
//            tooBig = !(D[i][0] < cutoff);
//            for (j = 1; j <= n; j++) {
//                htrans = Math.abs(i - j);
//                if (j > 1 && i > 1) {
//                    htrans *= 2;
//                }
//                dmin = D[i - 1][j - 1] + nu * htrans + D[i][j];
//
//                dist = Di1[i] + D[i - 1][j] + lambda + nu;
//                if (dmin > dist) {
//                    dmin = dist;
//                }
//                dist = Dj1[j] + D[i][j - 1] + lambda + nu;
//                if (dmin > dist) {
//                    dmin = dist;
//                }
//
//                D[i][j] = dmin;
//                if (tooBig && D[i][j] < cutoff) {
//                    tooBig = false;
//                }
//            }
//            //Early abandon
//            if (tooBig) {
//                return Double.POSITIVE_INFINITY;
//            }
//        }
//
//        dist = D[m][n];
////        for (i = 0; i <= m; i++){
////            StringBuilder a = new StringBuilder(D[i][0] + "");
////            for (j = 1; j <= n; j++){
////                a.append(",").append(D[i][j]);
////            }
////            System.out.println(a);
////        }
//        return dist;
//    }
//


    public double distance(double[] first, double[] second, double nu, double lambda) {
        final int m = first.length;
        final int n = second.length;
        final double[][] D = new double[m][n];

        double htrans;
        int i, j;
        final double nu_lambda = nu + lambda;
        final double nu2 = nu * 2;
        // border of the cost matrix initialization
        D[0][0] = dist(first[0], second[0]);
        for (i = 1; i < m; i++) D[i][0] = D[i - 1][0] + dist(first[i], first[i - 1]) + nu_lambda;
        for (j = 1; j < n; j++) D[0][j] = D[0][j - 1] + dist(second[j], second[j - 1]) + nu_lambda;

        for (i = 1; i < m; i++) {
            for (j = 1; j < n; j++) {
                htrans = Math.abs(i - j);
                double t = D[i - 1][j] + dist(first[i], first[i - 1]) + nu_lambda;
                double d = D[i - 1][j - 1] + dist(first[i], second[j]) + dist(first[i - 1], second[j - 1]) + nu2 * htrans;
                double p = D[i][j - 1] + dist(second[j], second[j - 1]) + nu_lambda;

                D[i][j] = GenericTools.min3(t, d, p);
            }
        }

        return D[m - 1][n - 1];
    }

    public double distance(double[] first, double[] second, double nu, double lambda, double cutoff) {
        boolean tooBig;
        final int m = first.length;
        final int n = second.length;
        final double[][] D = new double[m][n];

        double htrans;
        int i, j;
        final double nu_lambda = nu + lambda;
        final double nu2 = nu * 2;

        D[0][0] = dist(first[0], second[0]);
        for (i = 1; i < m; i++) D[i][0] = D[i - 1][0] + dist(first[i], first[i - 1]) + nu_lambda;
        for (j = 1; j < n; j++) D[0][j] = D[0][j - 1] + dist(second[j], second[j - 1]) + nu_lambda;

        for (i = 1; i < m; i++) {
            tooBig = !(D[i][0] < cutoff);
            for (j = 1; j < n; j++) {
                htrans = Math.abs(i - j);
                double t = D[i - 1][j] + dist(first[i], first[i - 1]) + nu_lambda;
                double d = D[i - 1][j - 1] + dist(first[i], second[j]) + dist(first[i - 1], second[j - 1]) + nu2 * htrans;
                double p = D[i][j - 1] + dist(second[j], second[j - 1]) + nu_lambda;

                D[i][j] = GenericTools.min3(t, d, p);
                if (tooBig && D[i][j] < cutoff) tooBig = false;
            }
            //Early abandon
            if (tooBig) return Double.POSITIVE_INFINITY;
        }

        return D[m - 1][n - 1];
    }
}
