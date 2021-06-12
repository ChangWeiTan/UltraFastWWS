package distances;

public class ElasticDistances {
    final static double EPSILON = 10e-12; // original: 10e-12, 10e-9 10e-15

    static double dist(double a, double b) {
        double d = a - b;
        return d * d;
    }
}
