package fastWWS.lazyAssessNN;

import datasets.Sequence;
import distances.classic.DTW;
import fastWWS.SequenceStatsCache;
import results.WarpingPathResults;

/**
 * LazyAssessNN with DTW
 */
public class LazyAssessNNDTW extends LazyAssessNN {
    private final DTW dtwComputer = new DTW();

    public LazyAssessNNDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public LazyAssessNNDTW(final Sequence query, final int index,
                           final Sequence reference, final int indexReference,
                           final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        tryLBKim();
        this.bestMinDist = minDist;
        this.status = LBStatus.LB_Kim;
    }


    public RefineReturnType tryToBeat(final double scoreToBeat, final int w) {
        setCurrentW(w);
        switch (status) {
            case Previous_LB_DTW:
            case Previous_DTW:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_KeoghQR;
                    else status = LBStatus.Full_LB_KeoghQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghQR;
            case Full_LB_KeoghQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.length()) status = LBStatus.Partial_LB_KeoghRQ;
                    else status = LBStatus.Full_LB_KeoghRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghRQ;
            case Full_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = dtwComputer.distanceExt(query.data[0], reference.data[0], currentW);
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w, final double bestSoFar) {
        setCurrentW(w);
        switch (status) {
            case Previous_LB_DTW:
            case Previous_DTW:
            case LB_Kim:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghQR:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghQR(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < query.length()) status = LBStatus.Partial_LB_KeoghQR;
                    else status = LBStatus.Full_LB_KeoghQR;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghQR;
            case Full_LB_KeoghQR:
                indexStoppedLB = 0;
                minDist = 0;
            case Partial_LB_KeoghRQ:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                tryContinueLBKeoghRQ(scoreToBeat);
                if (minDist > bestMinDist) bestMinDist = minDist;
                if (bestMinDist >= scoreToBeat) {
                    if (indexStoppedLB < reference.length()) status = LBStatus.Partial_LB_KeoghRQ;
                    else status = LBStatus.Full_LB_KeoghRQ;
                    return RefineReturnType.Pruned_with_LB;
                } else status = LBStatus.Full_LB_KeoghRQ;
            case Full_LB_KeoghRQ:
            case Partial_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = dtwComputer.distanceExt(query.data[0], reference.data[0], currentW, bestSoFar);
                if (res.earlyAbandon) {
                    minDist = bestSoFar;
                    minWindowValidity = currentW;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_DTW;
                    return RefineReturnType.Pruned_with_Dist;
                }
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
}
