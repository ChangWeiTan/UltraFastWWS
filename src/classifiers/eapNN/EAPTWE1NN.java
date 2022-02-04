package classifiers.eapNN;

import classifiers.classicNN.TWE1NN;
import datasets.Sequence;
import datasets.Sequences;
import distances.eap.EAPTWE;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNNEAP.LazyAssessNNEAPTWE;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for TWE-1NN
 * MSM-1NN with no lower bounds
 */
public class EAPTWE1NN extends TWE1NN {
    protected EAPTWE distComputer = new EAPTWE();

    public EAPTWE1NN() {
        this.classifierIdentifier = "TWE-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPTWE1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        this.classifierIdentifier = "TWE-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPTWE1NN(final int paramId, final Sequences trainData) {
        this.classifierIdentifier = "TWE-1NN_R1";
        this.setTrainingData(trainData);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.nu, this.lambda, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.nu, this.lambda, cutOffValue);
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

        final LazyAssessNNEAPTWE[] lazyAssessNNS = new LazyAssessNNEAPTWE[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNEAPTWE(cache);
        }
        final ArrayList<LazyAssessNNEAPTWE> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNEAPTWE d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = 0; paramId < nParams; ++paramId) {
                setParamsFromParamId(paramId);

                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNEAPTWE challenger = lazyAssessNNS[previous];
                        final LazyAssessNNEAPTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPTWE.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
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

                    for (LazyAssessNNEAPTWE challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNEAPTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPTWE.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            currPNN.set(previous, d, CandidateNN.Status.BC);
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
                        rrt = challenger.tryToBeat(toBeat, this.nu, this.lambda);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPTWE.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
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
                    double d = currPNN.distance;
                    int index = currPNN.nnIndex;
                    candidateNNS[paramId][current].set(index, d, CandidateNN.Status.NN);
                }
            }
        }
    }
}
