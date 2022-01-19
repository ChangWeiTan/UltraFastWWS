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

    public static double absDist(double a, double b) {
        Application.pointwiseCount++;
        return Math.abs(a - b);
    }

    public static double sqDist(double a, double b) {
        Application.pointwiseCount++;
        double d = a - b;
        return d * d;
    }

    public double l1(double[] a, double[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += absDist(a[i], b[i]);
        }
        return dist;
    }

    public double l1(double[] a, double[] b, double cutoff) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += absDist(a[i], b[i]);
            if (dist > cutoff) return cutoff;
        }
        return dist;
    }

    public double l2(double[] a, double[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += sqDist(a[i], b[i]);
        }
        return dist;
    }

    public double l2(double[] a, double[] b, double cutoff) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += sqDist(a[i], b[i]);
            if (dist > cutoff) return cutoff;
        }
        return dist;
    }
}
