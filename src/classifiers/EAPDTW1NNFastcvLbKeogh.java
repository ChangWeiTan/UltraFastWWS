package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * EAPDTW-1NN with FastCV training and Lb Keogh
 * No Upper bound
 */
public class EAPDTW1NNFastcvLbKeogh extends EAPDTW1NNLbKeogh {
    public EAPDTW1NNFastcvLbKeogh() {
        super();
        this.classifierIdentifier = "EAPDTW_1NN-FastCV-LbKeogh";
    }

    public EAPDTW1NNFastcvLbKeogh(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPDTW_1NN-FastCV-LbKeogh";
    }

    public EAPDTW1NNFastcvLbKeogh(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPDTW_1NN-FastCV-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
