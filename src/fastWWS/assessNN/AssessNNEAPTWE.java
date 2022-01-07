package fastWWS.assessNN;

import application.Application;
import datasets.Sequence;
import distances.eap.EAPTWE;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNN;

import static classifiers.classicNN.TWE1NN.tweLamdaParams;
import static classifiers.classicNN.TWE1NN.tweNuParams;

/**
 * LazyAssessNN with EAP TWE
 */
public class AssessNNEAPTWE extends LazyAssessNN {
    private final EAPTWE distComputer = new EAPTWE();
    protected double currentNu, currentLambda;

    public AssessNNEAPTWE(final SequenceStatsCache cache) {
        super(cache);
    }

    public AssessNNEAPTWE(final Sequence query, final int index,
                          final Sequence reference, final int indexReference,
                          final SequenceStatsCache cache) {
        super(query, index, reference, indexReference, cache);
        this.bestMinDist = minDist;
        this.status = LBStatus.None;
    }

    public void set(final Sequence query, final int index, final Sequence reference, final int indexReference) {
        // --- OTHER RESET
        indexStoppedLB = oldIndexStoppedLB = 0;
        currentNu = 0;
        currentLambda = 0;
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
    public void getUpperBound() {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
                Double.POSITIVE_INFINITY);
//        upperBoundDistance = distComputer.upperBoundDistance(query.data[0], reference.data[0],
//                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
//                Double.POSITIVE_INFINITY);
    }

    public void getUpperBound(int paramId) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[paramId / 10], tweLamdaParams[paramId % 10],
                Double.POSITIVE_INFINITY);
//        upperBoundDistance = distComputer.upperBoundDistance(query.data[0], reference.data[0],
//                tweNuParams[paramId / 10], tweLamdaParams[paramId % 10],
//                Double.POSITIVE_INFINITY);
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
                scoreToBeat);
//        upperBoundDistance = distComputer.upperBoundDistance(query.data[0], reference.data[0],
//                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
//                scoreToBeat);
    }

    public void getUpperBound(int paramId, final double scoreToBeat) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[paramId / 10], tweLamdaParams[paramId % 10],
                scoreToBeat);
    }

    private void setCurrentNuAndLambda(final double nu, final double lambda) {
        if (this.currentNu != nu) {
            this.currentLambda = lambda;
            this.currentNu = nu;
            this.minDist = 0.0;
            this.bestMinDist = minDist;
            this.status = LBStatus.None;
        } else if (this.currentLambda != lambda) {
            this.currentLambda = lambda;
            if (status == LBStatus.Full_TWE) {
                this.status = LBStatus.Previous_TWE;
            } else {
                this.status = LBStatus.None;
            }
        }
    }

    public RefineReturnType tryToBeat(final double scoreToBeat, final double nu, final double lambda, final double bestSoFar) {
        setCurrentNuAndLambda(nu, lambda);
        switch (status) {
            case None:
            case Previous_TWE:
            case Partial_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], nu, lambda, bestSoFar);
                if (minDist >= Double.MAX_VALUE) {
                    minDist = bestSoFar;
                    if (minDist > bestMinDist) bestMinDist = minDist;
                    status = LBStatus.Partial_TWE;
                    Application.eaCount++;
                    return RefineReturnType.Pruned_with_Dist;
                }
                Application.distCount++;
                if (minDist > bestMinDist) bestMinDist = minDist;
                status = LBStatus.Full_TWE;
            case Full_TWE:
                if (bestMinDist > scoreToBeat) return RefineReturnType.Pruned_with_Dist;
                else return RefineReturnType.New_best;
            default:
                throw new RuntimeException("Case not managed");
        }
    }

    @Override
    public double getDoubleValueForRanking() {
        double thisD = this.bestMinDist;

        // TWE
        switch (status) {
            case Full_TWE:
            case Full_LB_TWE:
            case Partial_TWE:
                return thisD / query.length();
            case Partial_LB_TWE:
                return thisD / indexStoppedLB;
            case Previous_TWE:
                return 0.8 * thisD / query.length();
            case Previous_LB_TWE:
                return thisD / oldIndexStoppedLB;
            case None:
                return Double.POSITIVE_INFINITY;
            default:
                throw new RuntimeException("shouldn't come here");
        }
    }
}
