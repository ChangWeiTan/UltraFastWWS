package fastWWS.lazyAssessNNEAP;

import application.Application;
import datasets.Sequence;
import distances.classic.LCSS;
import distances.eap.EAPLCSS;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;
import results.WarpingPathResults;

/**
 * LazyAssessNN with DTW
 */
public class LazyAssessNNEAPLCSS extends LazyAssessNN {
    private final EAPLCSS distComputer = new EAPLCSS();
    private int currentDelta;
    private double currentEpsilon;

    public LazyAssessNNEAPLCSS(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNNEAPLCSS(final Sequence query, final int index,
                               final Sequence reference, final int indexReference,
                               final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        minWindowValidity = 0;
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

    private void setCurrentDeltaAndEpsilon(final int delta, final double epsilon) {
        if (this.currentEpsilon != epsilon) {
            this.currentEpsilon = epsilon;
            this.currentDelta = delta;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.Previous_LB_LCSS;
        } else if (this.currentDelta != delta) {
            this.currentDelta = delta;
            if (status == LBStatus.Full_LCSS) {
                if (this.currentDelta < minWindowValidity) {
                    this.status = LBStatus.Previous_LCSS;
                }
            } else {
                this.status = LBStatus.Previous_LB_LCSS;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void tryFullLBLCSS() {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentDelta, currentEpsilon);
        final double[] UEQ = cache.getUE(indexQuery, currentDelta, currentEpsilon);
        double lcs = 0;
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (reference.value(index) <= UEQ[index] && reference.value(index) >= LEQ[index]) {
                lcs++;
            }
            indexStoppedLB++;
        }
        this.minDist = 1 - lcs / length;
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int delta, final double epsilon) {
        setCurrentDeltaAndEpsilon(delta, epsilon);
        switch (status) {
            case None:
            case Previous_LB_LCSS:
            case Previous_LCSS:
            case Partial_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryFullLBLCSS();
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_LCSS;
                    else status = LBStatus.Full_LB_LCSS;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_LCSS;
            case Full_LB_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentEpsilon, currentDelta, Double.POSITIVE_INFINITY);
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                Application.distCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_LCSS;
            case Full_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int delta, final double epsilon, final double bestSoFar) {
        setCurrentDeltaAndEpsilon(delta, epsilon);
        switch (status) {
            case None:
            case Partial_LCSS:
            case Previous_LB_LCSS:
            case Previous_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryFullLBLCSS();
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_LCSS;
                    else status = LBStatus.Full_LB_LCSS;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_LCSS;
            case Full_LB_LCSS:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentEpsilon, currentDelta, bestSoFar);
                if (res.earlyAbandon) {
                    Application.eaCount++;
                    minDist = bestSoFar;
                    minWindowValidity = 0;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_LCSS;
                    return RefineReturnType.Pruned_with_Dist;
                }
                Application.distCount++;
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_LCSS;
            case Full_LCSS:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // ERP
        switch (status) {
            case Full_LCSS:
            case Full_LB_LCSS:
            case Partial_LCSS:
                return thisD / query.length();
            case Partial_LB_LCSS:
                return thisD / indexStoppedLB;
            case Previous_LCSS:
                return 0.8 * thisD / query.length();
            case Previous_LB_LCSS:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
