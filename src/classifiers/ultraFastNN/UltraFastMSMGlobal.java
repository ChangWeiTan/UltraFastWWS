package classifiers.ultraFastNN;

import application.Application;
import classifiers.eapNN.EAPMSM1NN;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.AssessNNEAPMSM;
import results.TrainingClassificationResults;
import utils.EfficientSymmetricMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * UltraFastWWSearch
 * "Ultra fast warping window optimization for Dynamic Time Warping"
 */
public class UltraFastMSMGlobal extends EAPMSM1NN {
    final int ubParam = nParams - 1;

    public UltraFastMSMGlobal() {
        super();
        this.classifierIdentifier = "UltraFastMSM3";
    }

    public UltraFastMSMGlobal(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastMSM3";
    }

    public UltraFastMSMGlobal(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastMSM3";
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
        EfficientSymmetricMatrix upperBounds = new EfficientSymmetricMatrix(train.size());

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];

        final AssessNNEAPMSM[] lazyAssessNNS = new AssessNNEAPMSM[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new AssessNNEAPMSM(cache);
        }
        final ArrayList<AssessNNEAPMSM> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final AssessNNEAPMSM d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // ============================================================
            // do the largest window (full DTW) first
            // get paramid and initialise
            int paramId = 0;
            setParamsFromParamId(paramId);

            // sort existing series by their distances to their NN
            final CandidateNN[] nnArr = candidateNNS[paramId];
            int[] sortedIndices = IntStream.range(0, current)
                    .boxed().sorted((i, j) -> nnArr[j].compareTo(nnArr[i]))
                    .mapToInt(ele -> ele).toArray();

            // find the NN at full DTW.
            CandidateNN currPNN = nnArr[current];
            for (int i = 0; i < sortedIndices.length; i++) {
                int sortedIdx = sortedIndices[i];
                AssessNNEAPMSM challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[paramId][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                        challenger.getUpperBound();
                        bestSoFar = challenger.upperBoundDistance;
                        candidateNNS[ubParam][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                        upperBounds.put(current, previous, bestSoFar);
                    } else {
                        bestSoFar = candidateNNS[ubParam][current].distance;
                    }
                } else {
                    double a = upperBounds.get(current, currPNN.nnIndex);
                    if (a == 0) {
                        if (currPNN.nnIndex == candidateNNS[ubParam][current].nnIndex)
                            upperBounds.put(current, currPNN.nnIndex, candidateNNS[ubParam][current].distance);
                        else
                            upperBounds.put(current, currPNN.nnIndex,
                                    getUB(sCurrent.data[0], train.get(currPNN.nnIndex).data[0],
                                            candidateNNS[ubParam][current].distance));
                        a = upperBounds.get(current, currPNN.nnIndex);
                    }
                    if (prevNN.nnIndex >= 0) {
                        double b = upperBounds.get(previous, prevNN.nnIndex);
                        if (b == 0) {
                            if (prevNN.nnIndex == candidateNNS[ubParam][previous].nnIndex)
                                upperBounds.put(previous, prevNN.nnIndex, candidateNNS[ubParam][previous].distance);
                            else
                                upperBounds.put(previous, prevNN.nnIndex,
                                        getUB(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0],
                                                candidateNNS[ubParam][previous].distance));
                            b = upperBounds.get(previous, prevNN.nnIndex);
                        }
                        bestSoFar = Math.max(a, b);
                    } else {
                        bestSoFar = a;
                    }
                }
                AssessNNEAPMSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
            for (paramId = 1; paramId < nParams; paramId++) {
                setParamsFromParamId(paramId);

                currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int i = 0; i < current; ++i) {
                        final AssessNNEAPMSM challenger = challengers.get(i);
                        int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            if (candidateNNS[ubParam][previous].distance == Double.POSITIVE_INFINITY) {
                                challenger.getUpperBound();
                                bestSoFar = challenger.upperBoundDistance;
                                candidateNNS[ubParam][previous].set(current, bestSoFar, CandidateNN.Status.BC);
                                upperBounds.put(current, previous, bestSoFar);
                            } else {
                                bestSoFar = candidateNNS[ubParam][previous].distance;
                            }
                        } else {
                            double a = upperBounds.get(current, currPNN.nnIndex);
                            if (a == 0) {
                                if (currPNN.nnIndex == candidateNNS[ubParam][current].nnIndex)
                                    upperBounds.put(current, currPNN.nnIndex, candidateNNS[ubParam][current].distance);
                                else
                                    upperBounds.put(current, currPNN.nnIndex,
                                            getUB(sCurrent.data[0], train.get(currPNN.nnIndex).data[0],
                                                    candidateNNS[ubParam][current].distance));
                                a = upperBounds.get(current, currPNN.nnIndex);
                            }
                            if (prevNN.nnIndex >= 0) {
                                double b = upperBounds.get(previous, prevNN.nnIndex);
                                if (b == 0) {
                                    if (prevNN.nnIndex == candidateNNS[ubParam][previous].nnIndex)
                                        upperBounds.put(previous, prevNN.nnIndex, candidateNNS[ubParam][previous].distance);
                                    else
                                        upperBounds.put(previous, prevNN.nnIndex,
                                                getUB(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0],
                                                        candidateNNS[ubParam][previous].distance));
                                    b = upperBounds.get(previous, prevNN.nnIndex);
                                }
                                bestSoFar = Math.max(a, b);
                            } else {
                                bestSoFar = a;
                            }
                        }
                        final AssessNNEAPMSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                    AssessNNEAPMSM challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[paramId][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                            challenger.getUpperBound();
                            bestSoFar = challenger.upperBoundDistance;
                            candidateNNS[ubParam][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                            upperBounds.put(current, previous, bestSoFar);
                        } else {
                            bestSoFar = candidateNNS[ubParam][current].distance;
                        }
                    } else {
                        double a = upperBounds.get(current, currPNN.nnIndex);
                        if (a == 0) {
                            if (currPNN.nnIndex == candidateNNS[ubParam][current].nnIndex)
                                upperBounds.put(current, currPNN.nnIndex, candidateNNS[ubParam][current].distance);
                            else
                                upperBounds.put(current, currPNN.nnIndex,
                                        getUB(sCurrent.data[0], train.get(currPNN.nnIndex).data[0],
                                                candidateNNS[ubParam][current].distance));
                            a = upperBounds.get(current, currPNN.nnIndex);
                        }
                        if (prevNN.nnIndex >= 0) {
                            double b = upperBounds.get(previous, prevNN.nnIndex);
                            if (b == 0) {
                                if (prevNN.nnIndex == candidateNNS[ubParam][previous].nnIndex)
                                    upperBounds.put(previous, prevNN.nnIndex, candidateNNS[ubParam][previous].distance);
                                else
                                    upperBounds.put(previous, prevNN.nnIndex,
                                            getUB(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0],
                                                    candidateNNS[ubParam][previous].distance));
                                b = upperBounds.get(previous, prevNN.nnIndex);
                            }
                            bestSoFar = Math.max(a, b);
                        } else {
                            bestSoFar = a;
                        }
                    }
                    AssessNNEAPMSM.RefineReturnType rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                    rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                            if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                                challenger.getUpperBound();
                                bestSoFar = challenger.upperBoundDistance;
                                candidateNNS[ubParam][current].set(previous, bestSoFar, CandidateNN.Status.BC);
                                upperBounds.put(current, previous, bestSoFar);
                            } else {
                                bestSoFar = candidateNNS[ubParam][current].distance;
                            }
                        } else {
                            double a = upperBounds.get(current, currPNN.nnIndex);
                            if (a == 0) {
                                if (currPNN.nnIndex == candidateNNS[ubParam][current].nnIndex)
                                    upperBounds.put(current, currPNN.nnIndex, candidateNNS[ubParam][current].distance);
                                else
                                    upperBounds.put(current, currPNN.nnIndex,
                                            getUB(sCurrent.data[0], train.get(currPNN.nnIndex).data[0],
                                                    candidateNNS[ubParam][current].distance));
                                a = upperBounds.get(current, currPNN.nnIndex);
                            }
                            if (prevNN.nnIndex >= 0) {
                                double b = upperBounds.get(previous, prevNN.nnIndex);
                                if (b == 0) {
                                    if (prevNN.nnIndex == candidateNNS[ubParam][previous].nnIndex)
                                        upperBounds.put(previous, prevNN.nnIndex, candidateNNS[ubParam][previous].distance);
                                    else
                                        upperBounds.put(previous, prevNN.nnIndex,
                                                getUB(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0],
                                                        candidateNNS[ubParam][previous].distance));
                                    b = upperBounds.get(previous, prevNN.nnIndex);
                                }
                                bestSoFar = Math.max(a, b);
                            } else {
                                bestSoFar = a;
                            }
                        }

                        rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                        rrt = challenger.tryToBeat(toBeat, c, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPMSM.RefineReturnType.New_best) {
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
                    double lastD = candidateNNS[ubParam][current].distance;
                    index = currPNN.nnIndex;
                    if (d == lastD) {
                        for (int tmp = paramId; tmp < nParams; ++tmp) {
                            candidateNNS[tmp][current].set(index, d, CandidateNN.Status.NN);
                            classCounts[tmp][current] = classCounts[paramId][current].clone();
                        }
                    }
                }
            }
        }
    }

    private double getUB(double[] query, double[] reference, double cutoff) {
//        double a = distComputer.l1(query, reference, cutoff);
        double a = distComputer.distance(query, reference, msmParams[ubParam], cutoff);
        Application.ubCount++;
        if (a >= Double.MAX_VALUE) return cutoff;
        return a;
    }
}
