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
                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1]);
    }

    public void getUpperBound(int paramId) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[paramId / 10], tweLamdaParams[paramId % 10]);
    }

    public void getDiagUB() {
        upperBoundDistance = distComputer.diagonalDistance(query.data[0], reference.data[0]);
    }

    @Override
    public void getUpperBound(final double scoreToBeat) {
        upperBoundDistance = distComputer.distance(query.data[0], reference.data[0],
                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
                scoreToBeat);
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

    private void tryContinueLBTWED(final double scoreToBeat) {
        final int length = query.length();
        final double q0 = query.value(0);
        final double c0 = reference.value(0);
        double diff = q0 - c0;
        this.minDist = Math.min(diff * diff,
                Math.min(q0 * q0 + currentNu + currentLambda,
                        c0 * c0 + currentNu + currentLambda));
        this.indexStoppedLB = 1;
        while (indexStoppedLB < length && minDist <= scoreToBeat) {
            final int index = cache.getIndexNthHighestVal(indexReference, indexStoppedLB);
            if (index > 0) {
                final double curr = reference.value(index);
                final double prev = reference.value(index - 1);
                final double max = Math.max(cache.getMax(indexQuery), prev);
                final double min = Math.min(cache.getMin(indexQuery), prev);
                if (curr < min) {
                    diff = min - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                } else if (max < curr) {
                    diff = max - curr;
                    this.minDist += Math.min(currentNu, diff * diff);
                }
            }
            indexStoppedLB++;
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

    public RefineReturnType tryToBeat(final double scoreToBeat, final double nu, final double lambda) {
        setCurrentNuAndLambda(nu, lambda);
        switch (status) {
            case None:
            case Previous_TWE:
            case Partial_TWE:
                if (bestMinDist >= scoreToBeat) return RefineReturnType.Pruned_with_LB;
                minDist = distComputer.distance(query.data[0], reference.data[0], nu, lambda);
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
