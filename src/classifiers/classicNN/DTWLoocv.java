package classifiers.classicNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * DTW-1NN with LOOCV training and LbKeogh
 */
public class DTWLoocv extends DTW1NNLbKeogh {
    public DTWLoocv() {
        super();
        this.classifierIdentifier = "DTW_1NN-LOOCV-LbKeogh";
    }

    public DTWLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "DTW_1NN-LOOCV-LbKeogh";
    }

    public DTWLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "DTW_1NN-LOOCV-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocvLB(this.trainData);
    }
}
