package experiments;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import datasets.TimeSeriesDatasets;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.util.Objects;

import static application.Application.extractArguments;
import static utils.GenericTools.doTimeNs;
import static utils.GenericTools.println;

public class ScalabilityLength {
    static String moduleName = "ScalabilityLength";
    private static final String[] testArgs = new String[]{
            "-problem=HandOutlines",
            "-classifier=UltraFastWWSearch", // see classifiers in TimeSeriesClassifier.java
            "-paramId=99",
            "-cpu=-1",
            "-verbose=1",
            "-iter=0",
            "-eval=0",
            "-length=0",
    };

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        args = testArgs;
        extractArguments(args);
        if (Application.problem.equals(""))
            Application.problem = "HandOutlines";

        Application.printSummary(moduleName);

        switch (Application.problem) {
            case "all":
                for (String problem : TimeSeriesDatasets.longDatasets)
                    if (Application.scalabilityLengthRatio == 0) {
                        for (int i = 1; i <= 10; i++) {
                            singleRun(problem, 1.0 * i / 10);
                        }
                    } else {
                        singleRun(problem, Application.scalabilityLengthRatio);
                    }
                break;
            default:
                if (Application.scalabilityLengthRatio == 0) {
                    for (int i = 1; i <= 10; i++) {
                        singleRun(Application.problem, 1.0 * i / 10);
                    }
                } else {
                    singleRun(Application.problem, Application.scalabilityLengthRatio);
                }
                break;
        }

        final long endTime = System.nanoTime();
        println("[" + moduleName + "] Total time taken " + doTimeNs(endTime - startTime));
    }

    private static void singleRun(String problem, double ratio) throws Exception {
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

        println("[" + moduleName + "] Problem: " + problem + " -- " + ratio);
        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);

        trainData.chopSeries(ratio);

        println("[" + moduleName + "] Problem: " + problem + " -- " + trainData.length());

        if (Application.iteration == 0) {
            trainData.shuffle(0);
        }

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        if (Application.verbose > 1)
            println(classifier);

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        if (Application.verbose > 1)
            println("[" + moduleName + "]" + trainingResults);

        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm);
            testData.chopSeries(ratio);
            if (Application.iteration == 0) {
                testData.shuffle(0);
            }
            ClassificationResults classificationResults = classifier.evaluate(testData);
            classificationResults.problem = problem;
            if (Application.verbose > 1)
                println("[" + moduleName + "]" + classificationResults);

            double totalTime = trainingResults.elapsedTimeNanoSeconds + classificationResults.elapsedTimeNanoSeconds;
            if (Application.verbose > 1)
                println("[" + moduleName + "] Total time taken " + totalTime);

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
}
