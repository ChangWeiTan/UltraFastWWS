package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * DTW-1NN with FastCV training and Lb Keogh
 */
public class FastWWSearch extends DTW1NNLbKeogh {
    public FastWWSearch() {
        super();
        this.classifierIdentifier = "DTW_1NN-FastCV-LbKeogh";
    }

    public FastWWSearch(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "DTW_1NN-FastCV-LbKeogh";
    }

    public FastWWSearch(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "DTW_1NN-FastCV-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
