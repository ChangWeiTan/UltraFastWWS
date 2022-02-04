package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPERP;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;
import results.WarpingPathResults;

import static distances.classic.ERP.getWindowSize;

/**
 * LazyAssessNN with MSM
 */
public class AssessNNEAPERP extends LazyAssessNN {
    private final EAPERP distComputer = new EAPERP();
    private double currentG;
    private double currentBandSize;
    private int currentWindowSize;

    public AssessNNEAPERP(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNEAPERP(final Sequence query, final int index,
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
        currentG = 0;
        currentBandSize = 0;
        currentWindowSize = 0;
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
        nOperationsED = 0;
        indexStoppedED = 0;
        euclideanDistance = 0;

    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference,
                    final double bsf) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentG = 0;
        currentBandSize = 0;
        currentWindowSize = 0;
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
        nOperationsED = 0;
        indexStoppedED = 0;
        euclideanDistance = 0;
        getUpperBound(bsf);
    }

    private void setCurrentGandBandSize(final double g, final double bandSize) {
        if (this.currentG != g) {
            this.currentWindowSize = getWindowSize(query.length(), bandSize);
            this.currentBandSize = bandSize;
            this.currentG = g;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            indexStoppedLB = oldIndexStoppedLB = 0;
            this.status = LBStatus.None;
        } else if (this.currentBandSize != bandSize) {
            this.currentWindowSize = getWindowSize(query.length(), bandSize);
            this.currentBandSize = bandSize;
            if (status == LBStatus.Full_ERP) {
                if (this.currentBandSize < minWindowValidity) {
                    this.status = LBStatus.Previous_Band_ERP;
                }
            } else {
                this.status = LBStatus.None;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    private void tryContinueLBERPQR(final double scoreToBeat) {
        final int length = query.length();
        final double[] LEQ = cache.getLE(indexQuery, currentG, currentBandSize);
        final double[] UEQ = cache.getUE(indexQuery, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
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

    private void tryContinueLBERPRQ(final double scoreToBeat) {
        final int length = reference.length();
        final double[] LER = cache.getLE(indexReference, currentG, currentBandSize);
        final double[] UER = cache.getUE(indexReference, currentG, currentBandSize);
        this.minDist = 0.0;
        this.indexStoppedLB = 0;
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final double g, final double bandSize, final double bestSoFar) {
        setCurrentGandBandSize(g, bandSize);
        switch (status) {
            case None:
            case Previous_Band_ERP:
            case Partial_ERP:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_ERPQR:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                tryContinueLBERPQR(scoreToBeat);
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_ERPQR;
//                    else status = LBStatus.Full_LB_ERPQR;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_ERPQR;
//            case Full_LB_ERPQR:
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_ERPRQ:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                tryContinueLBERPRQ(scoreToBeat);
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < reference.length()) status = LBStatus.Partial_LB_ERPRQ;
//                    else status = LBStatus.Full_LB_ERPRQ;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_ERPRQ;
//            case Full_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentG, currentWindowSize, bestSoFar);
                if (res.earlyAbandon) {
                    Application.eaCount++;
                    minDist = bestSoFar;
                    minWindowValidity = 0;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_ERP;
                    return RefineReturnType.Pruned_with_Dist;
                }
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_ERP;
                Application.distCount++;
            case Full_ERP:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double g, final double bandSize) {
        setCurrentGandBandSize(g, bandSize);
        switch (status) {
            case None:
            case Previous_Band_ERP:
            case Partial_ERP:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_ERPQR:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                tryContinueLBERPQR(scoreToBeat);
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_ERPQR;
//                    else status = LBStatus.Full_LB_ERPQR;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_ERPQR;
//            case Full_LB_ERPQR:
//                indexStoppedLB = 0;
//                minDist = 0;
//            case Partial_LB_ERPRQ:
//                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
//                tryContinueLBERPRQ(scoreToBeat);
//                if (minDist > bestMinDist) bestMinDist = minDist;
//                if (bestMinDist >= scoreToBeat) {
//                    if (indexStoppedLB < reference.length()) status = LBStatus.Partial_LB_ERPRQ;
//                    else status = LBStatus.Full_LB_ERPRQ;
//                    return RefineReturnType.Pruned_with_LB;
//                } else status = LBStatus.Full_LB_ERPRQ;
//            case Full_LB_ERPRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = distComputer.distanceExt(query.data[0], reference.data[0], currentG, currentWindowSize, Double.POSITIVE_INFINITY);
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_ERP;
                Application.distCount++;
            case Full_ERP:
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
            case Full_ERP:
            case Full_LB_ERPQR:
            case Full_LB_ERPRQ:
            case Partial_ERP:
                return thisD / query.length();
            case Partial_LB_ERPQR:
            case Partial_LB_ERPRQ:
                return thisD / indexStoppedLB;
            case Previous_Band_ERP:
                return 0.8 * thisD / (query.length());
            case Previous_G_LB_ERP:
            case Previous_Band_LB_ERP:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
