package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import distances.lowerBounds.LbKeogh;
import results.TrainingClassificationResults;

/**
 * DTW-1NN with no training and Lb Keogh
 */
public class DTW1NNLbKeogh extends DTW1NN {
    protected LbKeogh lbComputer = new LbKeogh();

    public DTW1NNLbKeogh() {
        super();
        this.classifierIdentifier = this.classifierIdentifier + "-LbKeogh";
    }

    public DTW1NNLbKeogh(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = this.classifierIdentifier + "-LbKeogh";
    }

    public DTW1NNLbKeogh(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = this.classifierIdentifier + "-LbKeogh";
    }

    @Override
    public double lowerBound(final Sequence query,
                             final Sequence candidate,
                             final int queryIndex,
                             final int candidateIndex,
                             final double cutOffValue) {
        // U and L are built on query
        return lbComputer.distance(candidate,
                trainCache.getUE(queryIndex, window),
                trainCache.getLE(queryIndex, window),
                cutOffValue);
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv0LB(this.trainData);
    }

    @Override
    public int predict(final Sequence query) {
        int[] classCounts = new int[this.trainData.getNumClasses()];
        double[] U = new double[query.length()];
        double[] L = new double[query.length()];

        lbComputer.fillUL(query.data[0], this.window, U, L);

        double dist;

        Sequence candidate = trainData.get(0);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (int candidateIndex = 1; candidateIndex < trainData.size(); candidateIndex++) {
            candidate = trainData.get(candidateIndex);
            dist = this.lbComputer.distance(candidate, U, L, bsfDistance);

            if (dist < bsfDistance) {
                dist = distance(query, candidate, bsfDistance);
                if (dist < bsfDistance) {
                    bsfDistance = dist;
                    classCounts = new int[trainData.getNumClasses()];
                    classCounts[candidate.classificationLabel]++;
                } else if (dist == bsfDistance) {
                    classCounts[candidate.classificationLabel]++;
                }
            }
        }

        int bsfClass = -1;
        double bsfCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                bsfClass = i;
            }
        }
        return bsfClass;
    }
}
