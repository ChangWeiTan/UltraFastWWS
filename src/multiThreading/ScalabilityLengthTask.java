package multiThreading;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.util.Objects;
import java.util.concurrent.Callable;

import static utils.GenericTools.println;

public class ScalabilityLengthTask implements Callable<Integer> {
    String[] datasets;
    double[] subsetSizes;
    int threadCount;

    public ScalabilityLengthTask(String[] datasets, int threadCount) {
        this.datasets = datasets;
        this.threadCount = threadCount;
    }

    public ScalabilityLengthTask(String[] datasets, int threadCount, double[] subsetSize) {
        this.datasets = datasets;
        this.threadCount = threadCount;
        this.subsetSizes = subsetSize;
    }

    private void singleRun(String problem, double ratio) throws Exception {
        String outputPath = Objects.requireNonNullElseGet(Application.outputPath, () -> System.getProperty("user.dir") + "/outputs/scalability_length/");
        if (Application.paramId > 0)
            outputPath = outputPath +
                    Application.classifierName + "_" +
                    Application.paramId + "/" +
                    Application.iteration + "/" +
                    problem;
        else
            outputPath = outputPath +
                    Application.classifierName + "/" +
                    Application.iteration + "/" +
                    problem;

        println("[Thread_" + threadCount + "] Problem: " + problem + " -- " + ratio);
        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);

        trainData.chopSeries(ratio);

        println("[Thread_" + threadCount + "] Problem: " + problem + " -- " + trainData.length());

        if (Application.iteration == 0) {
            trainData.shuffle(0);
        }

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        if (Application.verbose > 1)
            println(classifier);

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        if (Application.verbose > 1)
            println("[Thread_" + threadCount + "]" + trainingResults);

        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm);
            testData.chopSeries(ratio);
            if (Application.iteration == 0) {
                testData.shuffle(0);
            }
            ClassificationResults classificationResults = classifier.evaluate(testData);
            classificationResults.problem = problem;
            if (Application.verbose > 1)
                println("[Thread_" + threadCount + "]" + classificationResults);

            double totalTime = trainingResults.elapsedTimeNanoSeconds + classificationResults.elapsedTimeNanoSeconds;
            if (Application.verbose > 1)
                println("[Thread_" + threadCount + "] Total time taken " + totalTime);

            Application.saveResults(
                    outputPath,
                    trainingResults,
                    classificationResults,
                    "results_" + ratio + ".csv");
        } else {
            Application.saveResults(
                    outputPath,
                    trainingResults,
                    "results_" + ratio + ".csv");
        }
    }

    @Override
    public Integer call() throws Exception {
        println("[Thread_" + threadCount + "] Datasets: " + datasets.length);

        for (int i = 0; i < datasets.length; i++) {
            singleRun(datasets[i], subsetSizes[i]);
        }
        println("[Thread_" + threadCount + "] Completed all datasets");
        return null;
    }
}
