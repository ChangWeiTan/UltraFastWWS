package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNUCRDTW;
import results.TrainingClassificationResults;

/**
 * DTW-1NN with LOOCV training and Lb Keogh
 */
public class UCRSuiteLOOCV extends DTW1NNLbKeogh {
    LazyAssessNNUCRDTW[][] assessNNUCRDTW;

    public UCRSuiteLOOCV() {
        super();
        this.classifierIdentifier = "UCRDTW_1NN-LOOCV-LbKeogh";
    }

    public UCRSuiteLOOCV(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UCRDTW_1NN-LOOCV-LbKeogh";
    }

    public UCRSuiteLOOCV(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UCRDTW_1NN-LOOCV-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        SequenceStatsCache cache = new SequenceStatsCache(trainData, trainData.length());
        assessNNUCRDTW = new LazyAssessNNUCRDTW[trainData.size()][trainData.size()];

        for (int i = 0; i < trainData.size(); i++) {
            for (int j = 0; j < trainData.size(); j++) {
                assessNNUCRDTW[i][j] = new LazyAssessNNUCRDTW(cache);
            }
        }
        return loocvLB(this.trainData);
    }

    @Override
    public int classifyLoocvLB(final Sequence query, final int queryIndex) {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;
        double bsfDistance = Double.POSITIVE_INFINITY;

        for (int candidateIndex = 0; candidateIndex < this.trainData.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            Sequence candidate = this.trainData.get(candidateIndex);
            assessNNUCRDTW[queryIndex][candidateIndex].set(query, queryIndex, candidate, candidateIndex);
            dist = assessNNUCRDTW[queryIndex][candidateIndex].tryToBeat(bsfDistance, this.window);

            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[this.trainData.getNumClasses()];
                classCounts[candidate.classificationLabel]++;
            } else if (dist == bsfDistance) {
                classCounts[candidate.classificationLabel]++;
            }
        }

        int bsfClass = -1;
        double bsfCount = -1;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > bsfCount) {
                bsfCount = classCounts[i];
                bsfClass = i;
            }
        }
        return bsfClass;
    }

}
