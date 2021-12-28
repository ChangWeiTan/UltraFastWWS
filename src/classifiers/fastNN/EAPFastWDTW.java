package classifiers.fastNN;

import classifiers.eapNN.EAPWDTW1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch EAPDTW-1NN with LbKeogh
 * Without early abandon
 */
public class EAPFastWDTW extends EAPWDTW1NN {
    public EAPFastWDTW() {
        super();
        this.classifierIdentifier = "EAPFastWDTW";
    }

    public EAPFastWDTW(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastWDTW";
    }

    public EAPFastWDTW(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastWDTW";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
