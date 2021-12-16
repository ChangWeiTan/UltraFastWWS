package fastWWS.assessNN;

import datasets.Sequence;
import distances.eap.EAPDTW;
import distances.eap.EAPWDTW;
import fastWWS.SequenceStatsCache;

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

    private void setCurrentWeightVector(final double[] weightVector) {
        this.currentWeightVector = weightVector;
        if (status == LBStatus.Full_WDTW) {
            this.status = LBStatus.Previous_WDTW;
        } else {
            this.status = LBStatus.None;
            this.oldIndexStoppedLB = indexStoppedLB;
        }
    }

    @Override
    public void getUpperBound() {
        // todo maybe we can use Euclidean instead of WDTW(g=0)
        // WDTW(q,c) @ g=0 = DTW(q,c)/2
        EAPDTW dtwComputer = new EAPDTW();
        upperBoundDistance = dtwComputer.distance(query.data[0], reference.data[0], query.length(), Double.POSITIVE_INFINITY) / 2;
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
        EAPDTW dtwComputer = new EAPDTW();
        upperBoundDistance = dtwComputer.distance(query.data[0], reference.data[0], query.length(), scoreToBeat) / 2;
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double[] weightVector, final double bestSoFar) {
        setCurrentWeightVector(weightVector);
        switch (status) {
            case None:
            case Previous_WDTW:
            case Partial_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                minDist = distComputer.distance(query.data[0], reference.data[0], weightVector, bestSoFar);
                if (minDist >= Double.MAX_VALUE) {
                    minDist = bestSoFar;
                    minWindowValidity = 0;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_WDTW;
                    return RefineReturnType.Pruned_with_Dist;
                }
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_WDTW;
            case Full_WDTW:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_Dist;
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
            case Partial_WDTW:
                return thisD / query.length();
            case Previous_WDTW:
                return 0.8 * thisD / query.length();
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
