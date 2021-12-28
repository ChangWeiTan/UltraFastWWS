package classifiers.fastNN;

import classifiers.eapNN.EAPDTW1NNLbKeogh;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch EAPDTW-1NN with LbKeogh
 * Without early abandon
 */
public class EAPFastWWSearch extends EAPDTW1NNLbKeogh {
    public EAPFastWWSearch() {
        super();
        this.classifierIdentifier = "EAPFastWWSearch-LbKeogh";
    }

    public EAPFastWWSearch(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastWWSearch-LbKeogh";
    }

    public EAPFastWWSearch(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastWWSearch-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
