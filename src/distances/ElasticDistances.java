package distances;

/**
 * Super class for Elastic Distances
 */
public class ElasticDistances {
    final static double EPSILON = 10e-12;

    static double dist(double a, double b) {
        double d = a - b;
        return d * d;
    }
}
