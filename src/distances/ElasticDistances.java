package distances;

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
        double d = a - b;
        return d * d;
    }
}
