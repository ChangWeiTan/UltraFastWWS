package fastWWS.assessNN;

import datasets.Sequence;
import fastWWS.SequenceStatsCache;

import static java.lang.Math.*;

/**
 * Super class for LazyAssessNN
 */
public abstract class LazyAssessNN implements Comparable<LazyAssessNN> {
    public int indexQuery;
    public int indexReference;         // Index for query and reference
    SequenceStatsCache cache;               // Cache to store the information for the sequences
    Sequence query, reference;              // Query and reference sequences
    int indexStoppedLB, oldIndexStoppedLB;  // Index where we stop LB
    int indexStoppedED;

    double minDist;                         // distance
    double bestMinDist;                     // best so far distance
    double kimDist;
    public double euclideanDistance;
    public double nnDistance;

    LBStatus status;                        // Status of Lower Bound

    protected int currentW;                           // Current warping window for DTW
    protected int minWindowValidity;                  // Minimum window validity for DTW, ERP, LCSS
    protected int nOperationsLBKim;                   // Number of operations for LB Kim
    protected int nOperationsED;

    public LazyAssessNN(final Sequence query, final int index,
                        final Sequence reference, final int indexReference,
                        final SequenceStatsCache cache) {
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
        this.minDist = 0;
        this.cache = cache;
        this.bestMinDist = minDist;
    }

    public LazyAssessNN(final SequenceStatsCache cache) {
        this.cache = cache;
    }

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
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
        nOperationsED = 0;
        indexStoppedED = 0;
        euclideanDistance = 0;
    }

    public void resetEuclidean() {
        indexStoppedED = 0;
        nOperationsED = 0;
        euclideanDistance = 0;
    }

    public void tryEuclidean() {
        while (indexStoppedED < query.length()) {
            final double diff = query.value(indexStoppedED) - reference.value(indexStoppedED);
            euclideanDistance += diff * diff;
            nOperationsED++;
            indexStoppedED++;
        }
    }

    public void tryEuclidean(final double scoreToBeat) {
        while (indexStoppedED < query.length() && minDist <= scoreToBeat) {
            final double diff = query.value(indexStoppedED) - reference.value(indexStoppedED);
            euclideanDistance += diff * diff;
            nOperationsED++;
            indexStoppedED++;
        }
    }

    protected void tryLBKim() {
        final double diffFirsts = query.value(0) - reference.value(0);
        final double diffLasts = query.value(query.length() - 1) - reference.value(reference.length() - 1);
        minDist = diffFirsts * diffFirsts + diffLasts * diffLasts;
        kimDist = minDist;
        nOperationsLBKim = 2;
        if (!cache.isMinFirst(indexQuery) && !cache.isMinFirst(indexReference) && !cache.isMinLast(indexQuery) && !cache.isMinLast(indexReference)) {
            final double diffMin = cache.getMin(indexQuery) - cache.getMin(indexReference);
            minDist += diffMin * diffMin;
            nOperationsLBKim++;
        }
        if (!cache.isMaxFirst(indexQuery) && !cache.isMaxFirst(indexReference) && !cache.isMaxLast(indexQuery) && !cache.isMaxLast(indexReference)) {
            final double diffMax = cache.getMax(indexQuery) - cache.getMax(indexReference);
            minDist += diffMax * diffMax;
            nOperationsLBKim++;
        }

        status = LBStatus.LB_Kim;
    }

    protected void tryContinueLBKeoghQR(final double scoreToBeat) {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentW);
        final double[] UEQ = cache.getUE(indexQuery, currentW);
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

    protected void tryContinueLBKeoghRQ(final double scoreToBeat) {
        final int length = reference.length();
        final double[] LER = cache.getLE(indexReference, currentW);
        final double[] UER = cache.getUE(indexReference, currentW);
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

    protected void tryFullLBKeoghQR() {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentW);
        final double[] UEQ = cache.getUE(indexQuery, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
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

    protected void tryFullLBKeoghRQ() {
        final int length = reference.length();
        final double[] LER = cache.getLE(indexReference, currentW);
        final double[] UER = cache.getUE(indexReference, currentW);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
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

    protected void tryContinueLBEnhanced(final double scoreToBeat) {
        final int length = query.length();
        final int l = length - 1;
        final int nBands = min(l / 2, 5);

        while (indexStoppedLB < nBands && minDist <= scoreToBeat) {
            final int rightEnd = l - indexStoppedLB;
            double minL = query.value(indexStoppedLB) - reference.value(indexStoppedLB);
            minL = abs(minL);
            double minR = query.value(rightEnd) - reference.value(rightEnd);
            minR = abs(minR);

            int rightStart;
            double tmp;
            for (int j = max(0, indexStoppedLB - currentW); j < indexStoppedLB; j++) {
                rightStart = l - j;
                tmp = query.value(indexStoppedLB) - reference.value(j);
                minL = min(minL, abs(tmp));
                tmp = query.value(j) - reference.value(indexStoppedLB);
                minL = min(minL, abs(tmp));

                tmp = query.value(rightEnd) - reference.value(rightStart);
                minR = min(minR, abs(tmp));
                tmp = query.value(rightStart) - reference.value(rightEnd);
                minR = min(minR, abs(tmp));
            }
            minDist += minL * minL + minR * minR;

            indexStoppedLB++;
        }
        if (minDist <= scoreToBeat) {
            final int lastIndex = l - nBands;
            final double[] LEQ = cache.getLE(indexQuery, currentW);
            final double[] UEQ = cache.getUE(indexQuery, currentW);
            while (indexStoppedLB < lastIndex && minDist <= scoreToBeat) {
                final double c = reference.value(indexStoppedLB);
                if (c < LEQ[indexStoppedLB]) {
                    final double diff = LEQ[indexStoppedLB] - c;
                    minDist += diff * diff;
                } else if (UEQ[indexStoppedLB] < c) {
                    final double diff = UEQ[indexStoppedLB] - c;
                    minDist += diff * diff;
                }
                indexStoppedLB++;
            }
        }
    }

    protected void setCurrentW(final int currentW) {
        if (this.currentW != currentW) {
            this.currentW = currentW;
            if (this.status == LBStatus.Full_DTW) {
                if (this.currentW < minWindowValidity) {
                    this.status = LBStatus.Previous_DTW;
                }
            } else {
                this.status = LBStatus.Previous_LB_DTW;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public void setBestMinDist(final double bestMinDist) {
        this.bestMinDist = bestMinDist;
    }

    public double getBestMinDist() {
        return this.bestMinDist;
    }

    @Override
    public String toString() {
        return "" + indexQuery + " - " + indexReference + " - " + bestMinDist;
    }

    public int getOtherIndex(final int index) {
        if (index == indexQuery) {
            return indexReference;
        } else {
            return indexQuery;
        }
    }

    public Sequence getSequenceForOtherIndex(final int index) {
        if (index == indexQuery) {
            return reference;
        } else {
            return query;
        }
    }

    public double getMinDist() {
        return minDist;
    }

    public void setMinDist(final double minDist) {
        this.minDist = minDist;
    }

    @Override
    public int compareTo(final LazyAssessNN o) {
        return this.compare(o);
    }

    private int compare(final LazyAssessNN o) {
        double num1 = this.getDoubleValueForRanking();
        double num2 = o.getDoubleValueForRanking();
        return Double.compare(num1, num2);
    }

    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // DTW
            case Full_DTW:
            case Full_LB_KeoghQR:
            case Full_LB_KeoghRQ:
            case Partial_DTW:
                return thisD / query.length();
            case LB_Kim:
                return thisD / nOperationsLBKim;
            case Partial_LB_KeoghQR:
            case Partial_LB_KeoghRQ:
                return thisD / indexStoppedLB;
            case Previous_DTW:
                return 0.8 * thisD / query.length();    // DTW(w+1) should be tighter
            case Previous_LB_DTW:
                if (indexStoppedLB == 0) {
                    //lb kim
                    return thisD / nOperationsLBKim;
                } else {
                    //lbkeogh
                    return thisD / oldIndexStoppedLB;
                }
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }

    @Override
    public boolean equals(final Object o) {
        LazyAssessNN d = (LazyAssessNN) o;
        return (this.indexQuery == d.indexQuery && this.indexReference == d.indexReference);
    }

    public LBStatus getStatus() {
        return status;
    }

    public double getBestLB() {
        return bestMinDist;
    }

    public Sequence getQuery() {
        return query;
    }

    public Sequence getReference() {
        return reference;
    }

    public double getDistance() {
        if (status == LBStatus.Full_WDTW) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is no valid already-computed Distance");
    }

    public double getDistance(final int window) {
        if ((status == LBStatus.Full_DTW) && minWindowValidity <= window) {
            return minDist;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is no valid already-computed Distance");
    }

    public int getMinWindowValidityForFullDistance() {
        if (status == LBStatus.Full_DTW) {
            return minWindowValidity;
        }
        throw new RuntimeException("Shouldn't call getDistance if not sure there is no valid already-computed Distance");
    }

    public int getMinwindow() {
        return minWindowValidity;
    }

    public void setMinwindow(final int w) {
        minWindowValidity = w;
    }

    public void setFullDistStatus() {
        this.status = LBStatus.Full_DTW;
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Internal types
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    public enum RefineReturnType {
        Pruned_with_LB, Pruned_with_Dist, New_best
    }

    public enum LBStatus {
        None, LB_Kim,
        Partial_LB_KeoghQR, Full_LB_KeoghQR, Partial_LB_KeoghRQ, Full_LB_KeoghRQ,   // DTW
        Partial_LB_Enhanced, Full_LB_Enhanced,
        Previous_LB_DTW, Previous_DTW, Full_DTW, Partial_DTW,                       // DTW
        Partial_LB_WDTWQR, Partial_LB_WDTWRQ, Full_LB_WDTWQR, Full_LB_WDTWRQ,       // WDTW
        Previous_LB_WDTW, Previous_WDTW, Full_WDTW,                                 // WDTW
    }
}