package classifiers;

import application.Application;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.AssessNNEAPWDTW;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * UltraFastWWSearch
 * "Ultra fast warping window optimization for Dynamic Time Warping"
 */
public class UltraFastWDTW extends EAPWDTW1NN {
    public UltraFastWDTW() {
        super();
        this.classifierIdentifier = "UltraFastWDTW";
    }

    public UltraFastWDTW(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastWDTW";
    }

    public UltraFastWDTW(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastWDTW";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }

    @Override
    public void initNNSTable(final Sequences train, final SequenceStatsCache cache) {
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is too small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        // initialise
        double bestSoFar;

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];
        boolean[] vectorCreated = new boolean[nParams];
        double[][] weightVectors = new double[nParams][maxWindow];

        final AssessNNEAPWDTW[] lazyAssessNNS = new AssessNNEAPWDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new AssessNNEAPWDTW(cache);
        }
        final ArrayList<AssessNNEAPWDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            if (Application.verbose > 1 && current % 10 == 0){
                System.out.print(".");
            }
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final AssessNNEAPWDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // ============================================================
            // do the largest window (full WDTW) first
            // get paramid and initialise
            int paramId = nParams - 1;
            setParamsFromParamId(paramId);
            if (!vectorCreated[paramId]) {
                initWeights(sCurrent.length());
                weightVectors[paramId] = weightVector;
                vectorCreated[paramId] = true;
            }

            // sort existing series by their distances to their NN
            final CandidateNN[] nnArr = candidateNNS[paramId];
            int[] sortedIndices = IntStream.range(0, current)
                    .boxed().sorted((i, j) -> nnArr[j].compareTo(nnArr[i]))
                    .mapToInt(ele -> ele).toArray();

            // find the NN at full DTW.
            CandidateNN currPNN = nnArr[current];
            for (int i = 0; i < sortedIndices.length; i++) {
                int sortedIdx = sortedIndices[i];
                AssessNNEAPWDTW challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[paramId][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    challenger.getUpperBound();
                    bestSoFar = challenger.upperBoundDistance;
                    candidateNNS[0][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                } else {
                    bestSoFar = Math.max(toBeat, prevNN.distance);
                }
                AssessNNEAPWDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
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
                rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
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
            classCounts[paramId][current] = classCounts[paramId][current].clone();

            // now sort the existing series based on distance at w+1
            Collections.sort(challengers);

            // remember the NN at w+1
            int nnAtPreviousWindow = 0;

            for (paramId = nParams - 2; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                if (!vectorCreated[paramId]) {
                    initWeights(sCurrent.length());
                    weightVectors[paramId] = weightVector;
                    vectorCreated[paramId] = true;
                }
                currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int i = 0; i < current; ++i) {
                        final AssessNNEAPWDTW challenger = challengers.get(i);
                        int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            challenger.getUpperBound();
                            bestSoFar = challenger.upperBoundDistance;
                            candidateNNS[0][previous].set(current, bestSoFar, CandidateNN.Status.BC);
                        } else {
                            bestSoFar = toBeat;
                        }
                        final AssessNNEAPWDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
                            d = challenger.getDistance();
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
                    AssessNNEAPWDTW challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[paramId][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        challenger.getUpperBound();
                        bestSoFar = challenger.upperBoundDistance;
                        candidateNNS[0][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                    } else {
                        bestSoFar = Math.max(toBeat, prevNN.distance);
                    }
                    AssessNNEAPWDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
                        d = challenger.getDistance();
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
                    rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
                        d = challenger.getDistance();
                        prevNN.set(current, d, CandidateNN.Status.NN);
                        if (d < toBeat) {
                            classCounts[paramId][previous] = new int[train.getNumClasses()];
                            classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                        } else if (d == toBeat) {
                            classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                        }
                    }

                    for (int i = 0; i < current; ++i) {
                        // skip the NN
                        if (i == nnAtPreviousWindow)
                            continue;

                        challenger = challengers.get(i);
                        previous = challenger.indexQuery;
                        prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        toBeat = currPNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            challenger.getUpperBound();
                            bestSoFar = challenger.upperBoundDistance;
                            candidateNNS[0][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                        } else {
                            bestSoFar = Math.max(toBeat, prevNN.distance);
                        }
                        rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
                            d = challenger.getDistance();
                            currPNN.set(previous, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                nnAtPreviousWindow = i;
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
                        rrt = challenger.tryToBeat(toBeat, weightVectors[paramId], bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPWDTW.RefineReturnType.New_best) {
                            d = challenger.getDistance();
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
                    d = currPNN.distance;
                    index = currPNN.nnIndex;
                    candidateNNS[paramId][current].set(index, d, CandidateNN.Status.NN);
                    classCounts[paramId][current] = classCounts[paramId][current].clone();
                }
            }
        }
        if (Application.verbose > 1) System.out.println("");
    }
}
