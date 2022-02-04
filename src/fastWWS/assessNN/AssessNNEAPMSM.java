package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPMSM;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;

import static classifiers.classicNN.MSM1NN.msmParams;

/**
 * LazyAssessNN with MSM
 */
public class AssessNNEAPMSM extends LazyAssessNN {
    private final EAPMSM distComputer = new EAPMSM();
    private double currentC;

    public AssessNNEAPMSM(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNEAPMSM(final Sequence query, final int index,
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
        currentC = 0;
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

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference,
                    final double bsf) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentC = 0;
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
        getUpperBound(bsf);
    }


    @Override
    public void getUpperBound() {
//        Application.ubCount++;
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[msmParams.length - 1], Double.POSITIVE_INFINITY);
    }

    public void getUpperBound(int paramId) {
//        Application.ubCount++;
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[paramId], Double.POSITIVE_INFINITY);
    }

    public void getDiagUB() {
//        Application.ubCount++;
        upperBoundDistance = distComputer.l1(query.data[0], reference.data[0]);
    }

    public void getDiagUB(final double scoreToBeat) {
//        Application.ubCount++;
        upperBoundDistance = distComputer.l1(query.data[0], reference.data[0], scoreToBeat);
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
//        Application.ubCount++;
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[msmParams.length - 1], scoreToBeat);
    }

    public void getUpperBound(final double scoreToBeat, int paramId) {
//        Application.ubCount++;
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[paramId + 1], scoreToBeat);
    }

    private void setCurrentC(final double c) {
        if (this.currentC != c) {
            this.currentC = c;
            if (status == LBStatus.Full_MSM) {
                this.status = LBStatus.Previous_MSM;
            } else {
                this.status = LBStatus.None;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void tryContinueLBMSM(final double scoreToBeat) {
        final int length = query.length();
        final double QMAX = cache.getMax(indexQuery);
        final double QMIN = cache.getMin(indexQuery);
        this.minDist = Math.abs(query.value(0) - reference.value(0));
        this.indexStoppedLB = 0;
        while (indexStoppedLB < length && minDist < scoreToBeat) {
            int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0 && index < length - 1) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                if (prev <= curr && curr < QMIN) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMIN), this.currentC);
                } else if (prev >= curr && curr >= QMAX) {
                    minDist += Math.min(Math.abs(reference.value(index) - QMAX), this.currentC);
                }
            }
            indexStoppedLB++;
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double c, final double bestSoFar) {
        setCurrentC(c);
        switch (status) {
            case None:
            case Previous_MSM:
            case Partial_MSM:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_MSM:
//                tryContinueLBMSM(scoreToBeat);
//                Application.lbCount++;
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_MSM;
//                    else status = LBStatus.Full_LB_MSM;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_MSM;
//            case Full_LB_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], currentC, bestSoFar);
                if (minDist >= Double.MAX_VALUE) {
                    minDist = bestSoFar;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_MSM;
                    Application.eaCount++;
                    return RefineReturnType.Pruned_with_Dist;
                }
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_MSM;
                Application.distCount++;
            case Full_MSM:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double c) {
        setCurrentC(c);
        switch (status) {
            case None:
            case Previous_MSM:
            case Partial_MSM:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_MSM:
//                tryContinueLBMSM(scoreToBeat);
//                Application.lbCount++;
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_MSM;
//                    else status = LBStatus.Full_LB_MSM;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_MSM;
//            case Full_LB_MSM:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], currentC);
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_MSM;
                Application.distCount++;
            case Full_MSM:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // MSM
            case Full_MSM:
            case Full_LB_MSM:
            case Partial_MSM:
                return thisD / query.length();
            case Partial_LB_MSM:
                return thisD / indexStoppedLB;
            case Previous_MSM:
                return 0.8 * thisD / query.length();
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
