package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPMSM;
import fastWWS.SequenceStatsCache;

import static classifiers.MSM1NN.msmParams;

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
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[msmParams.length - 1], Double.POSITIVE_INFINITY);
    }

    public void getUpperBound(int paramId) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[paramId], Double.POSITIVE_INFINITY);
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0], msmParams[msmParams.length - 1], scoreToBeat);
    }

    public void getUpperBound(final double scoreToBeat, int paramId) {
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final double c, final double bestSoFar) {
        setCurrentC(c);
        switch (status) {
            case None:
            case Previous_MSM:
            case Partial_MSM:
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

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        switch (status) {
            // MSM
            case Full_MSM:
            case Partial_MSM:
                return thisD / query.length();
            case Previous_MSM:
                return 0.8 * thisD / query.length();
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
