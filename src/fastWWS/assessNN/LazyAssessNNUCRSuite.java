package fastWWS.assessNN;

import datasets.Sequence;
import distances.DTW;
import fastWWS.SequenceStatsCache;

/**
 * LazyAssessNN with UCRSuite
 */
public class LazyAssessNNUCRSuite extends LazyAssessNN {
    private final DTW dtwComputer = new DTW();

    public LazyAssessNNUCRSuite(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNNUCRSuite(final Sequence query, final int index,
                                final Sequence reference, final int indexReference,
                                final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
    }

    @Override
    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentW = 0;
        minWindowValidity = 0;
        nOperationsLBKim = 0;
        // --- From constructor
        if (index < indexReference) {
            this.query = query;
            this.indexQuery = index;
            this.reference = reference;
            this.indexReference = indexReference;
        } else {
            this.query = reference;
            this.indexQuery = indexReference;
            this.reference = query;
            this.indexReference = index;
        }
        this.minDist = 0.0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    protected void tryLBKim(final double bsf) {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.length() - 1) - reference.value(reference.length() - 1);
        minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;
        if (minDist >= bsf)
            return;

        // 2 points at front
        double d0 = query.value(0) - reference.value(1);
        double d1 = query.value(1) - reference.value(0);
        double d2 = query.value(1) - reference.value(1);
        d0 = Math.min(d0 * d0, d1 * d1);
        d0 = Math.min(d0, d2 * d2);
        minDist += d0;
        if (minDist >= bsf)
            return;

        // 2 points at back
        d0 = query.value(query.length() - 1) - reference.value(reference.length() - 2);
        d1 = query.value(query.length() - 2) - reference.value(reference.length() - 1);
        d2 = query.value(query.length() - 2) - reference.value(reference.length() - 2);
        d0 = Math.min(d0 * d0, d1 * d1);
        d0 = Math.min(d0, d2 * d2);
        minDist += d0;
        if (minDist >= bsf)
            return;

        // 3 points at front
        d0 = query.value(0) - reference.value(2);
        d1 = query.value(1) - reference.value(2);
        d2 = query.value(2) - reference.value(2);
        double d3 = query.value(2) - reference.value(0);
        double d4 = query.value(2) - reference.value(1);
        d0 = Math.min(d0 * d0, d1 * d1);
        d0 = Math.min(d0, d2 * d2);
        d0 = Math.min(d0, d3 * d3);
        d0 = Math.min(d0, d4 * d4);
        minDist += d0;
        if (minDist >= bsf)
            return;

        // 3 points at back
        d0 = query.value(query.length() - 1) - reference.value(reference.length() - 3);
        d1 = query.value(query.length() - 2) - reference.value(reference.length() - 3);
        d2 = query.value(query.length() - 2) - reference.value(reference.length() - 3);
        d3 = query.value(query.length() - 3) - reference.value(reference.length() - 1);
        d4 = query.value(query.length() - 3) - reference.value(reference.length() - 2);
        d0 = Math.min(d0 * d0, d1 * d1);
        d0 = Math.min(d0, d2 * d2);
        d0 = Math.min(d0, d3 * d3);
        d0 = Math.min(d0, d4 * d4);
        minDist += d0;
    }

    protected void tryLBKeoghQR(final double scoreToBeat) {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentW);
        final double[] UEQ = cache.getUE(indexQuery, currentW);
        indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < LEQ[index]) {
                final double diff = LEQ[index] - c;
                minDist += diff * diff;
            } else if (UEQ[index] < c) {
                final double diff = UEQ[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    protected void tryLBKeoghRQ(final double scoreToBeat) {
        final int length = reference.length();
        final double[] LER = cache.getLE(indexReference, currentW);
        final double[] UER = cache.getUE(indexReference, currentW);
        indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < LER[index]) {
                final double diff = LER[index] - c;
                minDist += diff * diff;
            } else if (UER[index] < c) {
                final double diff = UER[index] - c;
                minDist += diff * diff;
            }
            indexStoppedLB++;
        }
    }

    public double tryToBeat(final double scoreToBeat, final int w) {
        setCurrentW(w);

        minDist = 0;
        tryLBKim(scoreToBeat);
        if (minDist > bestMinDist)
            bestMinDist = minDist;
        if (bestMinDist >= scoreToBeat)
            return Double.POSITIVE_INFINITY;

        minDist = 0;
        tryLBKeoghQR(scoreToBeat);
        if (minDist > bestMinDist)
            bestMinDist = minDist;
        if (bestMinDist >= scoreToBeat)
            return Double.POSITIVE_INFINITY;

        minDist = 0;
        tryLBKeoghRQ(scoreToBeat);
        if (minDist > bestMinDist)
            bestMinDist = minDist;
        if (bestMinDist >= scoreToBeat)
            return Double.POSITIVE_INFINITY;

        minDist = dtwComputer.distance(query.data[0], reference.data[0], currentW, scoreToBeat);
        if (minDist > bestMinDist) bestMinDist = minDist;

        if (bestMinDist >= scoreToBeat)
            return Double.POSITIVE_INFINITY;
        else
            return bestMinDist;
    }
}
