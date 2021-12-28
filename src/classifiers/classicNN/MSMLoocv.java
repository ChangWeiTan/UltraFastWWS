package classifiers.classicNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * MSM-1NN with LOOCV training
 */
public class MSMLoocv extends MSM1NN {
    public MSMLoocv() {
        super();
        this.classifierIdentifier = "MSM_1NN-LOOCV";
    }

    public MSMLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "MSM_1NN-LOOCV";
    }

    public MSMLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "MSM_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
