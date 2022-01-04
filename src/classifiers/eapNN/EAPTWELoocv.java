package classifiers.eapNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPTWE-1NN with LOOCV training
 */
public class EAPTWELoocv extends EAPTWE1NN {
    public EAPTWELoocv() {
        super();
        this.classifierIdentifier = "EAPTWE_1NN-LOOCV";
    }

    public EAPTWELoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPTWE_1NN-LOOCV";
    }

    public EAPTWELoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPTWE_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
