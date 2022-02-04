package classifiers.classicNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * ERP-1NN with LOOCV training
 */
public class ERPLoocv extends ERP1NN {
    public ERPLoocv() {
        super();
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    public ERPLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    public ERPLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "ERP_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
