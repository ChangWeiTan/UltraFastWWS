package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPDTW-1NN with LOOCV training and no lower bounds
 */
public class EAPDTW1NNLoocv extends EAPDTW1NN {

    public EAPDTW1NNLoocv() {
        super();
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV";
    }

    public EAPDTW1NNLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV";
    }

    public EAPDTW1NNLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
