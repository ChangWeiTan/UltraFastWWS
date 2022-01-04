package classifiers.eapFastNN;

import classifiers.eapNN.EAPERP1NN;
import classifiers.eapNN.EAPMSM1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch EAPMSM-1NN
 * Without early abandon
 */
public class EAPFastERP extends EAPERP1NN {
    public EAPFastERP() {
        super();
        this.classifierIdentifier = "EAPFastERP";
    }

    public EAPFastERP(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastERP";
    }

    public EAPFastERP(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastERP";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
