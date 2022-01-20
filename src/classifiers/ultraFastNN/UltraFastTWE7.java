package classifiers.ultraFastNN;

import classifiers.eapNN.EAPTWE1NN;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.AssessNNEAPTWE;
import results.TrainingClassificationResults;
import utils.EfficientSymmetricMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

import static distances.ElasticDistances.sqDist;

/**
 * UltraFastWWSearch
 * "Ultra fast warping window optimization for Dynamic Time Warping"
 */
public class UltraFastTWE7 extends EAPTWE1NN {
    public UltraFastTWE7() {
        super();
        this.classifierIdentifier = "UltraFastTWE7";
    }

    public UltraFastTWE7(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastTWE7";
    }

    public UltraFastTWE7(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastTWE7";
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
        final int nUb = 10;
        double bestSoFar;
        EfficientSymmetricMatrix upperBounds = new EfficientSymmetricMatrix(train.size());

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];

        final AssessNNEAPTWE[] lazyAssessNNS = new AssessNNEAPTWE[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new AssessNNEAPTWE(cache);
        }
        final ArrayList<AssessNNEAPTWE> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final AssessNNEAPTWE d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // ============================================================
            // do the largest window (full DTW) first
            // get paramid and initialise
            int paramId = 0;
            int nextUBParam = 9;
//            int ubCount = 0;
            setParamsFromParamId(paramId);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final AssessNNEAPTWE d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
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
                AssessNNEAPTWE challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[paramId][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    if (upperBounds.get(current, previous) == 0) {
                        challenger.getDiagUB();
                        bestSoFar = challenger.upperBoundDistance;
                        upperBounds.put(current, previous, bestSoFar);
                    } else {
                        bestSoFar = upperBounds.get(current, previous);
                    }
                } else {
                    double a = upperBounds.get(current, currPNN.nnIndex);
                    if (a == 0) {
                        if (currPNN.nnIndex == candidateNNS[nextUBParam][current].nnIndex)
                            upperBounds.put(current, currPNN.nnIndex, candidateNNS[nextUBParam][current].distance);
                        else
                            upperBounds.put(current, currPNN.nnIndex,
                                    diagonalUpperBound(sCurrent.data[0], train.get(currPNN.nnIndex).data[0]));
                        a = upperBounds.get(current, currPNN.nnIndex);
                    }
                    if (prevNN.nnIndex >= 0) {
                        double b = upperBounds.get(previous, prevNN.nnIndex);
                        if (b == 0) {
                            if (prevNN.nnIndex == candidateNNS[nextUBParam][previous].nnIndex)
                                upperBounds.put(previous, prevNN.nnIndex, candidateNNS[nextUBParam][previous].distance);
                            else
                                upperBounds.put(previous, prevNN.nnIndex,
                                        diagonalUpperBound(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0]));
                            b = upperBounds.get(previous, prevNN.nnIndex);
                        }
                        bestSoFar = Math.max(a, b);
                    } else {
                        bestSoFar = a;
                    }
                }
                AssessNNEAPTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                if (paramId % 10 == 0) {
                    nextUBParam = nextUBParam + 10;
                }

                setParamsFromParamId(paramId);
                currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int i = 0; i < current; ++i) {
                        final AssessNNEAPTWE challenger = challengers.get(i);
                        int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            if (upperBounds.get(current, previous) == 0) {
                                challenger.getDiagUB();
                                bestSoFar = challenger.upperBoundDistance;
                                upperBounds.put(current, previous, bestSoFar);
                            } else {
                                bestSoFar = upperBounds.get(current, previous);
                            }
                        } else {
                            double a = upperBounds.get(current, currPNN.nnIndex);
                            if (a == 0) {
                                if (currPNN.nnIndex == candidateNNS[nextUBParam][current].nnIndex)
                                    upperBounds.put(current, currPNN.nnIndex, candidateNNS[nextUBParam][current].distance);
                                else
                                    upperBounds.put(current, currPNN.nnIndex,
                                            diagonalUpperBound(sCurrent.data[0], train.get(currPNN.nnIndex).data[0]));
                                // update NNS
                                a = upperBounds.get(current, currPNN.nnIndex);
                                if (a < candidateNNS[nextUBParam][current].distance)
                                    candidateNNS[nextUBParam][current].set(currPNN.nnIndex, a, CandidateNN.Status.BC);
                            }
                            if (prevNN.nnIndex >= 0) {
                                double b = upperBounds.get(previous, prevNN.nnIndex);
                                if (b == 0) {
                                    if (prevNN.nnIndex == candidateNNS[nextUBParam][previous].nnIndex)
                                        upperBounds.put(previous, prevNN.nnIndex, candidateNNS[nextUBParam][previous].distance);
                                    else
                                        upperBounds.put(previous, prevNN.nnIndex,
                                                diagonalUpperBound(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0]));
                                    b = upperBounds.get(previous, prevNN.nnIndex);
                                    // update NNS
                                    if (b < candidateNNS[nextUBParam][previous].distance)
                                        candidateNNS[nextUBParam][previous].set(prevNN.nnIndex, b, CandidateNN.Status.BC);
                                }
                                bestSoFar = Math.max(a, b);
                            } else {
                                bestSoFar = a;
                            }
                        }
                        final AssessNNEAPTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                    AssessNNEAPTWE challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[paramId][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        if (upperBounds.get(current, previous) == 0) {
                            challenger.getDiagUB();
                            bestSoFar = challenger.upperBoundDistance;
                            upperBounds.put(current, previous, bestSoFar);
                        } else {
                            bestSoFar = upperBounds.get(current, previous);
                        }
                    } else {
                        double a = upperBounds.get(current, currPNN.nnIndex);
                        if (a == 0) {
                            if (currPNN.nnIndex == candidateNNS[nextUBParam][current].nnIndex)
                                upperBounds.put(current, currPNN.nnIndex, candidateNNS[nextUBParam][current].distance);
                            else
                                upperBounds.put(current, currPNN.nnIndex,
                                        diagonalUpperBound(sCurrent.data[0], train.get(currPNN.nnIndex).data[0]));
                            // update NNS
                            a = upperBounds.get(current, currPNN.nnIndex);
                            if (a < candidateNNS[nextUBParam][current].distance)
                                candidateNNS[nextUBParam][current].set(currPNN.nnIndex, a, CandidateNN.Status.BC);
                        }
                        if (prevNN.nnIndex >= 0) {
                            double b = upperBounds.get(previous, prevNN.nnIndex);
                            if (b == 0) {
                                if (prevNN.nnIndex == candidateNNS[nextUBParam][previous].nnIndex)
                                    upperBounds.put(previous, prevNN.nnIndex, candidateNNS[nextUBParam][previous].distance);
                                else
                                    upperBounds.put(previous, prevNN.nnIndex,
                                            diagonalUpperBound(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0]));
                                b = upperBounds.get(previous, prevNN.nnIndex);
                                // update NNS
                                if (b < candidateNNS[nextUBParam][previous].distance)
                                    candidateNNS[nextUBParam][previous].set(prevNN.nnIndex, b, CandidateNN.Status.BC);
                            }
                            bestSoFar = Math.max(a, b);
                        } else {
                            bestSoFar = a;
                        }
                    }
                    AssessNNEAPTWE.RefineReturnType rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                    rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                            if (upperBounds.get(current, previous) == 0) {
                                challenger.getDiagUB();
                                bestSoFar = challenger.upperBoundDistance;
                                upperBounds.put(current, previous, bestSoFar);
                            } else {
                                bestSoFar = upperBounds.get(current, previous);
                            }
                        } else {
                            double a = upperBounds.get(current, currPNN.nnIndex);
                            if (a == 0) {
                                if (currPNN.nnIndex == candidateNNS[nextUBParam][current].nnIndex)
                                    upperBounds.put(current, currPNN.nnIndex, candidateNNS[nextUBParam][current].distance);
                                else
                                    upperBounds.put(current, currPNN.nnIndex,
                                            diagonalUpperBound(sCurrent.data[0], train.get(currPNN.nnIndex).data[0]));
                                // update NNS
                                a = upperBounds.get(current, currPNN.nnIndex);
                                if (a < candidateNNS[nextUBParam][current].distance)
                                    candidateNNS[nextUBParam][current].set(currPNN.nnIndex, a, CandidateNN.Status.BC);
                            }
                            if (prevNN.nnIndex >= 0) {
                                double b = upperBounds.get(previous, prevNN.nnIndex);
                                if (b == 0) {
                                    if (prevNN.nnIndex == candidateNNS[nextUBParam][previous].nnIndex)
                                        upperBounds.put(previous, prevNN.nnIndex, candidateNNS[nextUBParam][previous].distance);
                                    else
                                        upperBounds.put(previous, prevNN.nnIndex,
                                                diagonalUpperBound(train.get(previous).data[0], train.get(prevNN.nnIndex).data[0]));
                                    b = upperBounds.get(previous, prevNN.nnIndex);
                                    // update NNS
                                    if (b < candidateNNS[nextUBParam][previous].distance)
                                        candidateNNS[nextUBParam][previous].set(prevNN.nnIndex, b, CandidateNN.Status.BC);
                                }
                                bestSoFar = Math.max(a, b);
                            } else {
                                bestSoFar = a;
                            }
                        }
                        rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
                        rrt = challenger.tryToBeat(toBeat, nu, lambda, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPTWE.RefineReturnType.New_best) {
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
    }

    private double getUB(double[] query, double[] reference, double cutoff) {
        double a = distComputer.distance(query, reference,
                tweNuParams[tweNuParams.length - 1], tweLamdaParams[tweLamdaParams.length - 1],
                cutoff);
        if (a >= Double.MAX_VALUE) return cutoff;
        return a;
    }

    private double getUB(double[] query, double[] reference, double cutoff, int paramId) {
        double a;
        if (cutoff > Double.MAX_VALUE)
            a = distComputer.distance(query, reference,
                    tweNuParams[paramId / 10], tweLamdaParams[paramId % 10]);
        else {
            a = distComputer.distance(query, reference,
                    tweNuParams[paramId / 10], tweLamdaParams[paramId % 10],
                    cutoff);
            if (a >= Double.MAX_VALUE) return cutoff;
        }
        return a;
    }

    public double diagonalUpperBound(double[] lines, double[] cols) {
        final int m = lines.length;
        double dist = sqDist(lines[0], cols[0]);

        for (int i = 1; i < m; i++) {
            dist += sqDist(lines[i], cols[i]) + sqDist(lines[i - 1], cols[i - 1]);
        }

        return dist;
    }
}
