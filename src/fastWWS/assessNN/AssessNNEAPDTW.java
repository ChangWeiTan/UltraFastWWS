package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPDTW;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;
import results.WarpingPathResults;

/**
 * AssessNN with EAP
 * No lowerbounds
 */
public class AssessNNEAPDTW extends LazyAssessNN {
    private final EAPDTW dtwComputer = new EAPDTW();

    public AssessNNEAPDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNEAPDTW(final Sequence query, final int index,
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
        this.minDist = 0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
        nOperationsED = 0;
        indexStoppedED = 0;
        euclideanDistance = 0;
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference,
                    final double bsf) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = indexStoppedED = 0;
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
        this.minDist = 0;
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
        nOperationsED = 0;
        indexStoppedED = 0;
        euclideanDistance = 0;
        tryEuclidean(bsf);
    }

    @Override
    public void setCurrentW(final int currentW) {
        if (this.currentW != currentW) {
            this.currentW = currentW;
            if (this.status == LBStatus.Full_DTW) {
                if (this.currentW < minWindowValidity) {
                    this.status = LBStatus.Previous_DTW;
                }
            } else {
                this.status = LBStatus.None;
                this.oldIndexStoppedLB = indexStoppedLB;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w, final double bestSoFar) {
        setCurrentW(w);
        switch (status) {
            case None:
            case Previous_DTW:
            case Partial_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                final WarpingPathResults res = dtwComputer.distanceExt(
                        query.data[0],
                        reference.data[0],
                        currentW,
                        bestSoFar);
                if (res.earlyAbandon) {
                    Application.eaCount++;
                    minDist = bestSoFar;
                    minWindowValidity = 0;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_DTW;
                    return RefineReturnType.Pruned_with_Dist;
                }
                Application.distCount++;
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
            case Full_DTW:
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
            // DTW
            case Full_DTW:
            case Partial_DTW:
                return thisD / query.length();
            case Previous_DTW:
                return 0.8 * thisD / query.length();    // DTW(w+1) should be tighter
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }

    public RefineReturnType tryDistance(final int w, final double bestSoFar) {
        setCurrentW(w);
        final WarpingPathResults res = dtwComputer.distanceExt(
                query.data[0],
                reference.data[0],
                currentW,
                bestSoFar);
        if (res.earlyAbandon) {
            minDist = bestSoFar;
            minWindowValidity = currentW;
            if (minDist > bestMinDist) bestMinDist = minDist;
            status = LBStatus.Partial_DTW;
            return RefineReturnType.Pruned_with_Dist;
        }
        minDist = res.distance;
        if (minDist > bestMinDist) bestMinDist = minDist;
        status = LBStatus.Full_DTW;
        minWindowValidity = res.distanceFromDiagonal;
        return RefineReturnType.New_best;
    }

//    @Override
//    public int getMinWindowValidityForFullDistance() {
//        if (status == LBStatus.Full_DTW || status == LBStatus.Partial_DTW) {
//            return minWindowValidity;
//        }
//        throw new RuntimeException("Shouldn't call getDistance if not sure there is no valid already-computed Distance");
//    }
//
//    @Override
//    public double getDistance(final int window) {
//        if ((status == LBStatus.Full_DTW || status == LBStatus.Partial_DTW) && minWindowValidity <= window) {
//            return minDist;
//        }
//        throw new RuntimeException("Shouldn't call getDistance if not sure there is no valid already-computed Distance");
//    }
}
