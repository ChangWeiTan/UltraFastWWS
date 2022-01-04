package classifiers.eapFastNN;

import classifiers.eapNN.EAPTWE1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class EAPFastTWE extends EAPTWE1NN {
    public EAPFastTWE() {
        super();
        this.classifierIdentifier = "EAPFastTWE";
    }

    public EAPFastTWE(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastTWE";
    }

    public EAPFastTWE(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastTWE";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
