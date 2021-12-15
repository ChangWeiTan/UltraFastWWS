package fastWWS.assessNN;

import datasets.Sequence;
import distances.classic.DTW;
import fastWWS.SequenceStatsCache;
import results.WarpingPathResults;

/**
 * AssessNN with DTW
 * No lowerbounds
 */
public class AssessNNDTW extends LazyAssessNN {
    private final DTW dtwComputer = new DTW();

    public AssessNNDTW(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNDTW(final Sequence query, final int index,
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
        // Number of operations for LB Kim
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

    @Override
    protected void setCurrentW(final int currentW) {
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w) {
        setCurrentW(w);
        switch (status) {
            case None:
            case Previous_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                final WarpingPathResults res = dtwComputer.distanceExt(
                        query.data[0],
                        reference.data[0],
                        currentW);
                minDist = res.distance;
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
            case Full_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final int w, final double bestSoFar) {
        setCurrentW(w);
        switch (status) {
            case None:
            case Previous_DTW:
            case Partial_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
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
                minWindowValidity = res.distanceFromDiagonal;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_DTW;
            case Full_DTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
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
}
