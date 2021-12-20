package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch EAPMSM-1NN
 * Without early abandon
 */
public class EAPFastMSM extends EAPMSM1NN {
    public EAPFastMSM() {
        super();
        this.classifierIdentifier = "EAPFastMSM";
    }

    public EAPFastMSM(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastMSM";
    }

    public EAPFastMSM(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastMSM";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
