package distances.classic;

import distances.ElasticDistances;
import results.WarpingPathResults;
import utils.GenericTools;

public class ED extends ElasticDistances {
    public static double distance(final double[] first, final double[] second) {
        final int n = first.length;
        final int m = second.length;
        double d = 0;
        int i = 0;
        while (i < Math.min(n, m)) {
            final double diff = first[i] - second[i];
            d += diff * diff;
            i++;
        }
        return d;
    }
}
