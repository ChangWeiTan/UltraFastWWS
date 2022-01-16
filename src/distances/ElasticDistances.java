package distances;

import application.Application;

/**
 * Super class for Elastic Distances
 */
public class ElasticDistances {
    public enum Measures {
        DTW,
        WDTW,
        MSM,
        ERP,
        TWE
    }

    public final static double EPSILON = 10e-12;

    public static double dist(double a, double b) {
        Application.pointwiseCount++;
        double d = a - b;
        return d * d;
    }

    public double l1(double[] a, double[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += Math.abs(a[i] - b[i]);
        }
        return dist;
    }
}
