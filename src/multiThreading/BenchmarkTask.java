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

public class BenchmarkTask implements Callable<Integer> {
    String[] datasets;
    int threadCount;

    public BenchmarkTask(String[] datasets, int threadCount) {
        this.datasets = datasets;
        this.threadCount = threadCount;
    }

    private void singleRun(String problem) throws Exception {
        String outputPath = Objects.requireNonNullElseGet(Application.outputPath, () -> System.getProperty("user.dir") + "/outputs/benchmark/");
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

        println("[Thread_" + threadCount + "] Problem: " + problem);

        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);
        Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm);
        if (Application.verbose > 1) {
            trainData.summary();
            testData.summary();
        }

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        if (Application.verbose > 1)
            println(classifier);

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        if (Application.verbose > 1)
            println("[Thread_" + threadCount + "]" + trainingResults);
        else if (Application.verbose == 1)
            println("[Thread_" + threadCount + "] Problem: " + problem + ", Time: " + trainingResults.doTimeNs());
        if (Application.doEvaluation) {
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
                    classificationResults);
        } else {
            Application.saveResults(
                    outputPath,
                    trainingResults);
        }
    }

    @Override
    public Integer call() throws Exception {
        println("[Thread_" + threadCount + "] Datasets: " + datasets.length);
        for (String dataset : datasets) {
            singleRun(dataset);
        }
        println("[Thread_" + threadCount + "] Completed all datasets");
        return null;
    }
}
