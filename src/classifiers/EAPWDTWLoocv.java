package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPWDTW-1NN with LOOCV training
 */
public class EAPWDTWLoocv extends EAPWDTW1NN {
    public EAPWDTWLoocv() {
        super();
        this.classifierIdentifier = "EAPWDTW_1NN-LOOCV";
    }

    public EAPWDTWLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPWDTW_1NN-LOOCV";
    }

    public EAPWDTWLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPWDTW_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
