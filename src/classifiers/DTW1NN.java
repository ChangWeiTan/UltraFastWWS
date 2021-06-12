package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import distances.DTW;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNDTW;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for DTW-1NN
 * DTW-1NN with no lower bounds
 */
public class DTW1NN extends OneNearestNeighbour {
    protected double r;
    protected int window;
    protected DTW distComputer = new DTW();

    public DTW1NN() {
        this.classifierIdentifier = "DTW-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public DTW1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        this.r = 1;
        this.window = trainData.length();
        this.classifierIdentifier = "DTW-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public DTW1NN(final int paramId, final Sequences trainData) {
        this.r = 1;
        this.window = trainData.get(0).length();
        this.classifierIdentifier = "DTW-1NN_R1";
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
                "\n[CLASSIFIER SUMMARY] r: " + r +
                "\n[CLASSIFIER SUMMARY] window: " + window +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        if (r < 1) {
            window = distComputer.getWindowSize(Math.max(first.length(), second.length()), r);
            return distComputer.distance(first.data[0], second.data[0], window);
        }
        return distComputer.distance(first.data[0], second.data[0]);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        if (r < 1) {
            window = distComputer.getWindowSize(Math.max(first.length(), second.length()), r);
            return distComputer.distance(first.data[0], second.data[0], window, cutOffValue);
        }
        return distComputer.distance(first.data[0], second.data[0], cutOffValue);
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

        final LazyAssessNNDTW[] lazyAssessNNS = new LazyAssessNNDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNDTW(cache);
        }
        final ArrayList<LazyAssessNNDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                final int win = distComputer.getWindowSize(maxWindow, r);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNDTW challenger = lazyAssessNNS[previous];
                        final LazyAssessNNDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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

                    for (LazyAssessNNDTW challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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
                        rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
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
                    final int winEnd = getParamIdFromWindow(r, train.length());
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                    }
                }
            }
        }
    }

    @Override
    public void setTrainingData(final Sequences trainData) {
        this.trainData = trainData;
        this.trainCache = new SequenceStatsCache(trainData, trainData.get(0).length());
        this.window = distComputer.getWindowSize(this.trainData.get(0).length(), this.r);
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (paramId < 100 && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        r = 1.0 * paramId / 100;
        window = distComputer.getWindowSize(trainData.get(0).length(), r);
    }

    @Override
    public String getParamInformationString() {
        return "r=" + this.r;
    }

    protected int getParamIdFromWindow(final int w, final int n) {
        double r = 1.0 * w / n;
        return (int) Math.ceil(r * 100);
    }
}
