package classifiers.classicNN;

import datasets.Sequence;
import datasets.Sequences;
import distances.classic.LCSS;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNNLCSS;
import results.TrainingClassificationResults;
import utils.GenericTools;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for LCSS-1NN
 * MSM-1NN with no lower bounds
 */
public class LCSS1NN extends OneNearestNeighbour {
    // parameters
    protected int delta;
    protected double epsilon;
    protected double[] epsilons;
    protected int[] deltas;
    protected boolean epsilonsAndDeltasRefreshed;

    protected LCSS distComputer = new LCSS();

    public LCSS1NN() {
        this.classifierIdentifier = "LCSS-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public LCSS1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        this.classifierIdentifier = "LCSS-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public LCSS1NN(final int paramId, final Sequences trainData) {
        this.classifierIdentifier = "LCSS-1NN_R1";
        this.setTrainingData(trainData);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public void summary() {
        System.out.println(toString());
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] delta: " + this.delta +
                "\n[CLASSIFIER SUMMARY] epsilon: " + this.epsilon +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.epsilon, this.delta);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.epsilon, this.delta);
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv0(this.trainData);
    }

    /**
     * Code from "Efficient search of the best warping window for dynamic time warping"
     */
    @Override
    public void initNNSTable(final Sequences train, final SequenceStatsCache cache) {
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is too small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];

        final LazyAssessNNLCSS[] lazyAssessNNS = new LazyAssessNNLCSS[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNLCSS(cache);
        }
        final ArrayList<LazyAssessNNLCSS> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNLCSS d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);

                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNLCSS challenger = lazyAssessNNS[previous];
                        final LazyAssessNNLCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNNLCSS challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNLCSS.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.getNumClasses()];
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, this.delta, this.epsilon);

                        // --- Check the result
                        if (rrt == LazyAssessNNLCSS.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(this.delta);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    final int r = currPNN.r;
                    final double d = currPNN.distance;
                    final int index = currPNN.nnIndex;
                    final double prevEpsilon = epsilon;
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevEpsilon == epsilon && delta >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(tmp);
                    }
                }
            }
        }
    }

    @Override
    public void setTrainingData(final Sequences trainData) {
        this.trainData = trainData;
        this.trainCache = new SequenceStatsCache(trainData, trainData.get(0).length());
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (paramId < 100 && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        if (!epsilonsAndDeltasRefreshed) {
            double stdTrain = GenericTools.stdv_p(trainData);
            double stdFloor = stdTrain * 0.2;
            epsilons = GenericTools.getInclusive10(stdFloor, stdTrain);
            deltas = GenericTools.getInclusive10(0, (trainData.length()) / 4);
            epsilonsAndDeltasRefreshed = true;
        }
        this.delta = deltas[paramId % 10];
        this.epsilon = epsilons[paramId / 10];
    }

    @Override
    public String getParamInformationString() {
        return "delta=" + this.delta + ", epsilon=" + this.epsilon;
    }

    protected int getParamIdFromWindow(final int currentParamId, final int r) {
                int i = currentParamId;
        while (i >= 0 && deltas[i % 10] != r)
            i--;

        return i;
    }
}
