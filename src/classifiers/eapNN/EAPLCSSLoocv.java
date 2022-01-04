package classifiers.eapNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * ERP-1NN with LOOCV training
 */
public class EAPLCSSLoocv extends EAPLCSS1NN {
    public EAPLCSSLoocv() {
        super();
        this.classifierIdentifier = "EAPLCSS_1NN-LOOCV";
    }

    public EAPLCSSLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPLCSS_1NN-LOOCV";
    }

    public EAPLCSSLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPLCSS_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
