package classifiers.classicNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * TWE-1NN with LOOCV training
 */
public class TWELoocv extends TWE1NN {
    public TWELoocv() {
        super();
        this.classifierIdentifier = "TWE_1NN-LOOCV";
    }

    public TWELoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "TWE_1NN-LOOCV";
    }

    public TWELoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "TWE_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
