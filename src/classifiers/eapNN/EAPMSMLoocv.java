package classifiers.eapNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPWDTW-1NN with LOOCV training
 */
public class EAPMSMLoocv extends EAPMSM1NN {
    public EAPMSMLoocv() {
        super();
        this.classifierIdentifier = "EAPMSM_1NN-LOOCV";
    }

    public EAPMSMLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPMSM_1NN-LOOCV";
    }

    public EAPMSMLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPMSM_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
