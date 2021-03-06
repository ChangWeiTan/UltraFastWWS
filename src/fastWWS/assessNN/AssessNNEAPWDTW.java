package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPWDTW;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;

/**
 * AssessNN with EAP
 * No lowerbounds
 */
public class AssessNNEAPWDTW extends LazyAssessNN {
    private final EAPWDTW distComputer = new EAPWDTW();
    protected double[] currentWeightVector;           // weight vector for WDTW

    public AssessNNEAPWDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNEAPWDTW(final Sequence query, final int index,
                           final Sequence reference, final int indexReference,
                           final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    @Override
    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
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
        this.minDist = 0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
        indexStoppedED = 0;
        upperBoundDistance = 0;
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference,
                    final double bsf) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
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
        this.minDist = 0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
        indexStoppedED = 0;
        upperBoundDistance = 0;
        getUpperBound(bsf);
    }

    @Override
    public void getUpperBound() {
        tryEuclidean();
        upperBoundDistance = euclideanDistance / 2;
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
        tryEuclidean(scoreToBeat);
        upperBoundDistance = euclideanDistance / 2;
    }

    private void setCurrentWeightVector(final double[] weightVector) {
        this.currentWeightVector = weightVector;
        if (status == LBStatus.Full_WDTW) {
            this.status = LBStatus.Previous_WDTW;
        } else {
            this.status = LBStatus.None;
            this.oldIndexStoppedLB = indexStoppedLB;
        }
    }

    private void tryContinueLBWDTWQR(final double scoreToBeat) {
        final int length = query.length();
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            final double c = reference.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    private void tryContinueLBWDTWRQ(final double scoreToBeat) {
        final int length = reference.length();
        final double QMAX = cache.getMax(indexReference);
        final double QMIN = cache.getMin(indexReference);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexQuery, indexStoppedLB);
            final double c = query.value(index);
            if (c < QMIN) {
                final double diff = QMIN - c;
                minDist += diff * diff * currentWeightVector[0];
            } else if (QMAX < c) {
                final double diff = QMAX - c;
                minDist += diff * diff * currentWeightVector[0];
            }
            indexStoppedLB++;
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double[] weightVector) {
        setCurrentWeightVector(weightVector);
        switch (status) {
            case None:
            case Previous_WDTW:
            case Partial_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBWDTWQR(scoreToBeat);
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWQR;
                    else status = LBStatus.Full_LB_WDTWQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWQR;
            case Full_LB_WDTWQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBWDTWRQ(scoreToBeat);
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWRQ;
                    else status = LBStatus.Full_LB_WDTWRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWRQ;
            case Full_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                minDist = distComputer.distance(query.data[0], reference.data[0], weightVector);
                Application.distCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double[] weightVector, final double bestSoFar) {
        setCurrentWeightVector(weightVector);
        switch (status) {
            case None:
            case Previous_WDTW:
            case Partial_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBWDTWQR(scoreToBeat);
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWQR;
                    else status = LBStatus.Full_LB_WDTWQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWQR;
            case Full_LB_WDTWQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBWDTWRQ(scoreToBeat);
                Application.lbCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_WDTWRQ;
                    else status = LBStatus.Full_LB_WDTWRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_WDTWRQ;
            case Full_LB_WDTWRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                minDist = distComputer.distance(query.data[0], reference.data[0], weightVector, bestSoFar);
                if (minDist >= Double.MAX_VALUE) {
                    Application.eaCount++;
                    minDist = bestSoFar;
                    minWindowValidity = 0;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_WDTW;
                    return RefineReturnType.Pruned_with_Dist;
                }
                Application.distCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // WDTW
        switch (status) {
            case Full_WDTW:
            case Full_LB_WDTWQR:
            case Full_LB_WDTWRQ:
            case Partial_WDTW:
                return thisD / query.length();
            case Partial_LB_WDTWQR:
            case Partial_LB_WDTWRQ:
                return thisD / indexStoppedLB;
            case Previous_WDTW:
                return 0.8 * thisD / query.length();
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
