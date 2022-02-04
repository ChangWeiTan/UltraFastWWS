package classifiers.eapNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPDTW-1NN with LOOCV training and LbKeogh
 */
public class EAPLoocv extends EAPDTW1NNLbKeogh {
    public EAPLoocv() {
        super();
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV-LbKeogh";
    }

    public EAPLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV-LbKeogh";
    }

    public EAPLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPDTW_1NN-LOOCV-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocvLB(this.trainData);
    }
}
