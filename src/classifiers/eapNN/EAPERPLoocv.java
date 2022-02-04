package classifiers.eapNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * ERP-1NN with LOOCV training
 */
public class EAPERPLoocv extends EAPERP1NN {
    public EAPERPLoocv() {
        super();
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    public EAPERPLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    public EAPERPLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
