package classifiers;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import results.TrainingClassificationResults;

/**
 * A super class for 1 NN classifier
 */
public abstract class OneNearestNeighbour extends TimeSeriesClassifier {
    SequenceStatsCache trainCache; // cache for training set

    final int nParams = 100;
    int maxWindow;
    int[][][] classCounts;

    CandidateNN[][] candidateNNS;

    public abstract double distance(final Sequence first,
                                    final Sequence second) throws Exception;

    public abstract double distance(final Sequence first,
                                    final Sequence second,
                                    final double cutOffValue) throws Exception;

    public double lowerBound(final Sequence query,
                             final Sequence candidate,
                             final int queryIndex,
                             final int candidateIndex,
                             final double cutOffValue) {
        // default to 0 if no lower bounds
        return 0;
    }

    public abstract void initNNSTable(final Sequences trainData, final SequenceStatsCache cache) throws Exception;

    @Override
    public int predict(final Sequence query) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        Sequence candidate = trainData.get(0);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (int candidateIndex = 1; candidateIndex < trainData.size(); candidateIndex++) {
            candidate = trainData.get(candidateIndex);
            dist = distance(query, candidate);
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[trainData.getNumClasses()];
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

    /**
     * Do Leave-One-Out CV with the default parameter ID
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv0(final Sequences train) throws Exception {
        double[] accAndPreds;
        double bsfAcc = -1;
        final double[] predictions = new double[train.size()];

        if (Application.verbose > 1)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        final long start = System.nanoTime();
        accAndPreds = loocvAccAndPreds(train, this.bestParamId);
        if (accAndPreds[0] > bsfAcc) {
            bsfAcc = accAndPreds[0];
            System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
        }
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions
        );
        results.paramId = bestParamId;

        if (Application.verbose > 1)
            System.out.printf("[1-NN] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV with the default parameter ID and lower bound
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv0LB(final Sequences train) throws Exception {
        double[] accAndPreds;
        double bsfAcc = -1;
        final double[] predictions = new double[train.size()];

        if (Application.verbose > 1)
            System.out.println("[1-NN] LOOCV0 for " + this.classifierIdentifier);

        final long start = System.nanoTime();
        accAndPreds = loocvAccAndPredsLB(train, bestParamId);
        if (accAndPreds[0] > bsfAcc) {
            bsfAcc = accAndPreds[0];
            System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
        }
        final long end = System.nanoTime();
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;

        if (Application.verbose > 1)
            System.out.printf("[1-NN] LOOCV0 Results: %s, Acc=%.5f, Time=%s%n",
                    getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocv(final Sequences train) throws Exception {
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];
        bestParamId = -1;
        double bsfAcc = -1;
        double[] predictions = new double[train.size()];

        if (Application.verbose > 1) {
            System.out.print("[1-NN] LOOCV for " + this.classifierIdentifier + ", training");
            System.out.print("loocv_acc = [");
        }

        final long start = System.nanoTime();

        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = loocvAccAndPreds(train, paramId);

            if (Application.verbose > 1)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bestParamId = paramId;
                bsfAcc = accAndPreds[0];
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 1)
            System.out.printf("[1-NN] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do Leave-One-Out CV with lower bound
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults loocvLB(final Sequences train) throws Exception {
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];

        bestParamId = -1;
        double bsfAcc = -1;
        double[] predictions = new double[train.size()];

        if (Application.verbose > 1) {
            System.out.print("[1-NN] LOOCV for " + this.classifierIdentifier + ", training");
            System.out.print("loocv_acc = [");
        }

        final long start = System.nanoTime();
        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = loocvAccAndPredsLB(train, paramId);

            if (Application.verbose > 1)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bestParamId = paramId;
                bsfAcc = accAndPreds[0];
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier,
                bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 1)
            System.out.printf("[1-NN] LOOCV Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    /**
     * Do FastWWSearch
     * Code from "Efficient search of the best warping window for dynamic time warping"
     *
     * @param train training dataset
     * @return training results
     */
    public TrainingClassificationResults fastWWSearch(final Sequences train) throws Exception {
        bestParamId = -1;
        double bsfAcc = -1;
        double[] accAndPreds;
        int[] cvParams = new int[nParams];
        double[] cvAcc = new double[nParams];
        double[] predictions = new double[train.size()];
        this.maxWindow = train.length();

        if (Application.verbose > 1)
            System.out.print("[1-NN] Fast Parameter Search for " + this.classifierIdentifier + ", training ");

        final long start = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("[1-NN] Initialising NNs table for Fast Parameter Search");

        initNNSTable(train, trainCache);

        for (int paramId = 0; paramId < nParams; paramId++) {
            cvParams[paramId] = paramId;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = fastWWSearchAccAndPred(train, paramId, train.size());

            if (Application.verbose > 1)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[paramId] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bestParamId = paramId;
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier, bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 1)
            System.out.printf("[1-NN] Fast Parameter Search Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    protected double[] loocvAccAndPreds(final Sequences train, final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.size() + 1];
        for (int i = 0; i < train.size(); i++) {
            final Sequence query = train.get(i);
            actual = query.classificationLabel;
            pred = this.classifyLoocv(query, i);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.size();

        return accAndPreds;
    }

    protected double[] loocvAccAndPredsLB(final Sequences train, final int paramId) throws Exception {
        this.setParamsFromParamId(paramId);

        int correct = 0;
        double pred, actual;

        double[] accAndPreds = new double[train.size() + 1];
        for (int i = 0; i < train.size(); i++) {
            final Sequence query = train.get(i);
            actual = query.classificationLabel;
            pred = this.classifyLoocvLB(query, i);
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = (double) correct / train.size();

        return accAndPreds;
    }

    protected double[] fastWWSearchAccAndPred(final Sequences train, final int paramId, final int n) {
        this.setParamsFromParamId(paramId);
        int correct = 0;
        double pred, actual;

        final double[] accAndPreds = new double[n + 1];
        for (int i = 0; i < n; i++) {
            actual = train.get(i).classificationLabel;
            pred = -1;
            double bsfCount = -1;
            for (int c = 0; c < classCounts[paramId][i].length; c++) {
                if (classCounts[paramId][i][c] > bsfCount) {
                    bsfCount = classCounts[paramId][i][c];
                    pred = c;
                }
            }
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = 1.0 * correct / n;

        return accAndPreds;
    }

    public int classifyLoocv(final Sequence query, final int queryIndex) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        int candidateIndex = (queryIndex > 0) ? 0 : 1;
        int nextIndex = candidateIndex + 1;
        Sequence candidate = trainData.get(candidateIndex);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (candidateIndex = nextIndex; candidateIndex < trainData.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            candidate = trainData.get(candidateIndex);
            dist = distance(query, candidate, bsfDistance);
            if (dist < bsfDistance) {
                bsfDistance = dist;
                classCounts = new int[trainData.getNumClasses()];
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

    public int classifyLoocvLB(final Sequence query, final int queryIndex) throws Exception {
        int[] classCounts = new int[this.trainData.getNumClasses()];

        double dist;

        int candidateIndex = (queryIndex > 0) ? 0 : 1;
        int nextIndex = candidateIndex + 1;
        Sequence candidate = this.trainData.get(candidateIndex);
        double bsfDistance = distance(query, candidate);
        classCounts[candidate.classificationLabel]++;

        for (candidateIndex = nextIndex; candidateIndex < this.trainData.size(); candidateIndex++) {
            if (queryIndex == candidateIndex)
                continue;
            candidate = this.trainData.get(candidateIndex);
            dist = lowerBound(query, candidate, queryIndex, candidateIndex, bsfDistance);
            if (dist < bsfDistance) {
                dist = distance(query, candidate, bsfDistance);
                if (dist < bsfDistance) {
                    bsfDistance = dist;
                    classCounts = new int[this.trainData.getNumClasses()];
                    classCounts[candidate.classificationLabel]++;
                } else if (dist == bsfDistance) {
                    classCounts[candidate.classificationLabel]++;
                }
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
