package application;

import classifiers.*;
import classifiers.FastWWSearch;
import datasets.Sequences;
import fileIO.OutFile;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import java.io.File;

public class Application {
    public static Runtime runtime = Runtime.getRuntime();   // get the run time of the application
    public static final String optionsSep = "=";
    public static String outputPath;
    public static String datasetPath;
    public static String problem = "";
    public static String classifierName = "DTW-1NN";
    public static int paramId = 0;
    public static int verbose = 0;
    public static boolean znorm = true;
    public static boolean retrain = true;
    public static int iteration = 0;
    public static int numThreads = 0;
    public static double scalabilityTrainRatio = 0;
    public static double scalabilityLengthRatio = 0;
    public static boolean doEvaluation = true;
    private final static String defaultSaveFilename = "results.csv";

    public static void extractArguments(final String[] args) throws Exception {
        /**
         * Extract arguments from command line
         */
        System.out.print("[APP] Input arguments:");
        for (String arg : args) {
            final String[] options = arg.trim().split(optionsSep);
            System.out.print(" " + arg);
            if (options.length >= 2)
                switch (options[0]) {
                    case "-out":
                        outputPath = options[1];
                        break;
                    case "-data":
                        datasetPath = options[1];
                        break;
                    case "-problem":
                        problem = options[1];
                        break;
                    case "-paramId":
                        paramId = Integer.parseInt(options[1]);
                        break;
                    case "-znorm":
                        znorm = Boolean.parseBoolean(options[1]);
                        break;
                    case "-cpu":
                        numThreads = Integer.parseInt(options[1]);
                        break;
                    case "-classifier":
                        classifierName = options[1];
                        break;
                    case "-iter":
                        iteration = Integer.parseInt(options[1]);
                        break;
                    case "-trainSize":
                        scalabilityTrainRatio = Double.parseDouble(options[1]);
                        break;
                    case "-length":
                        scalabilityLengthRatio = Double.parseDouble(options[1]);
                        break;
                    case "-verbose":
                        verbose = Integer.parseInt(options[1]);
                        break;
                    case "-eval":
                        doEvaluation = Boolean.parseBoolean(options[1]);
                        break;
                    default:
                        throw new Exception("Incorrect arguments, check your arguments");
                }
            else
                throw new Exception("Incorrect arguments, check your arguments");
        }
        System.out.println();

        // reset parameter ID if we are looking for it
        if (classifierName.toLowerCase().contains("loocv") || classifierName.toLowerCase().contains("fastcv"))
            paramId = -1;

        if (Application.datasetPath == null) {
            String osName = System.getProperty("os.name");
            String username = System.getProperty("user.name");
            if (osName.contains("Window")) {
                Application.datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
            } else {
                Application.datasetPath = "/home/" + username + "/workspace/Dataset/UCRArchive_2018/";
            }
        }
    }

    /**
     * Initialise TSC classifier
     */
    public static TimeSeriesClassifier initTSC(final Sequences trainData) {
        TimeSeriesClassifier classifier;
        switch (classifierName) {
            /// WDTW distance
            case "UltraFastWDTW":
                classifier = new UltraFastWDTW(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPFastWDTWEA":
                classifier = new EAPFastWDTWEA(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPFastWDTW":
                classifier = new EAPFastWDTW(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "FastWDTW":
                classifier = new FastWDTW(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPWDTWLOOCV":
                classifier = new EAPWDTWLoocv(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCV;
                break;
            case "WDTWLOOCV":
                classifier = new WDTWLoocv(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCV;
                break;

            /// DTW distance
            case "UltraFastWWSearchFull":
                // UltraFastWWSearch with sorting the training set in descending order and then sorting on full DTW
                classifier = new UltraFastWWSearchFULL(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "UltraFastWWSearchV2":
                // UltraFastWWSearch with sorting the training set in descending order and then sorting on full DTW
                classifier = new UltraFastWWSearchV2(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "UltraFastWWSearchV1":
                // UltraFastWWSearch with tighter upper bound for early abandoning
                classifier = new UltraFastWWSearchV1(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPFastWWSearchEA":
                // FastWWSearch with EAP and simple upper bound for early abandoning
                classifier = new EAPFastWWSearchEA(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPFastWWSearchNoLb":
                // FastWWSearch with EAP without lower bounds
                classifier = new EAPFastWWSearchNoLb(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPFastWWSearch":
                // FastWWSearch with EAP with lower bounds and no early abandoning
                classifier = new EAPFastWWSearch(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "EAPLOOCV":
                classifier = new EAPLoocv(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCVLB;
                break;

            case "UCRSuiteLOOCV":
                // LOOCV with UCRSuite
                classifier = new UCRSuiteLOOCV(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCVLB;
                break;

            case "FastWWSearchEA":
                classifier = new FastWWSearchEA(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "FastWWSearchNoLb":
                // Naive DTW without upper bounding and no lower bound in NNs
                classifier = new FastWWSearchNoLb(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "FastWWSearch":
                classifier = new FastWWSearch(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
            case "DTWLOOCV":
                classifier = new DTWLoocv(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.LOOCVLB;
                break;

            default:
                // UltraFastWWSearch
                classifier = new UltraFastWWSearch(paramId, trainData);
                classifier.trainingOptions = TimeSeriesClassifier.TrainOpts.FastWWS;
                break;
        }

        return classifier;
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   ClassificationResults classificationResults) throws Exception {
        saveResults(outputPath,
                trainingClassificationResults,
                classificationResults,
                defaultSaveFilename);
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults) throws Exception {
        saveResults(outputPath,
                trainingClassificationResults,
                defaultSaveFilename);
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   ClassificationResults classificationResults,
                                   String filename) throws Exception {
        OutFile outFile = new OutFile(outputPath, filename);
        outFile.writeLine("problem," + trainingClassificationResults.problem);
        outFile.writeLine("classifier," + trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0)
            outFile.writeLine("paramId," + trainingClassificationResults.paramId);

        outFile.writeLine("train_acc," + trainingClassificationResults.accuracy);
        outFile.writeLine("train_correct," + trainingClassificationResults.nbCorrect);
        outFile.writeLine("train_size," + trainingClassificationResults.trainSize);
        outFile.writeLine("train_time," + trainingClassificationResults.doTimeNs());
        outFile.writeLine("train_time_ns," + trainingClassificationResults.elapsedTimeNanoSeconds);
        outFile.writeLine("train_dtw_count," + trainingClassificationResults.dtwCount);
        outFile.writeLine("train_ea_count," + trainingClassificationResults.eaCount);

        outFile.writeLine("test_acc," + classificationResults.accuracy);
        outFile.writeLine("test_correct," + classificationResults.nbCorrect);
        outFile.writeLine("test_size," + classificationResults.testSize);
        outFile.writeLine("test_time," + classificationResults.doTimeNs());
        outFile.writeLine("test_time_ns," + classificationResults.elapsedTimeNanoSeconds);

        StringBuilder str = new StringBuilder("[");
        for (int i = 0; i < classificationResults.confMat.length; i++) {
            str.append("[").append(classificationResults.confMat[i][0]);
            for (int j = 1; j < classificationResults.confMat[i].length; j++) {
                str.append(":").append(classificationResults.confMat[i][j]);
            }
            str.append("]");
        }
        str.append("]");
        outFile.writeLine("test_conf_mat," + str);

        if (trainingClassificationResults.cvParams != null) {
            str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(":").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            outFile.writeLine("cv_param," + str);

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(":").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            outFile.writeLine("cv_acc," + str);
        }

        str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
        for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
            str.append(":").append(trainingClassificationResults.predictions[i]);
        }
        str.append("]");
        outFile.writeLine("train_predictions," + str);

        str = new StringBuilder("[" + classificationResults.predictions[0]);
        for (int i = 1; i < classificationResults.predictions.length; i++) {
            str.append(":").append(classificationResults.predictions[i]);
        }
        str.append("]");
        outFile.writeLine("test_predictions," + str);

        outFile.closeFile();
    }

    public static void saveResults(String outputPath,
                                   TrainingClassificationResults trainingClassificationResults,
                                   String filename) throws Exception {
        OutFile outFile = new OutFile(outputPath, filename);
        outFile.writeLine("problem," + trainingClassificationResults.problem);
        outFile.writeLine("classifier," + trainingClassificationResults.classifier);
        if (trainingClassificationResults.paramId >= 0)
            outFile.writeLine("paramId," + trainingClassificationResults.paramId);

        outFile.writeLine("train_acc," + trainingClassificationResults.accuracy);
        outFile.writeLine("train_correct," + trainingClassificationResults.nbCorrect);
        outFile.writeLine("train_size," + trainingClassificationResults.trainSize);
        outFile.writeLine("train_time," + trainingClassificationResults.doTimeNs());
        outFile.writeLine("train_time_ns," + trainingClassificationResults.elapsedTimeNanoSeconds);
        outFile.writeLine("train_dtw_count," + trainingClassificationResults.dtwCount);
        outFile.writeLine("train_ea_count," + trainingClassificationResults.eaCount);

        if (trainingClassificationResults.cvParams != null) {
            StringBuilder str = new StringBuilder("[" + trainingClassificationResults.cvParams[0]);
            for (int i = 1; i < trainingClassificationResults.cvParams.length; i++) {
                str.append(":").append(trainingClassificationResults.cvParams[i]);
            }
            str.append("]");
            outFile.writeLine("cv_param," + str);

            str = new StringBuilder("[" + trainingClassificationResults.cvAcc[0]);
            for (int i = 1; i < trainingClassificationResults.cvAcc.length; i++) {
                str.append(":").append(trainingClassificationResults.cvAcc[i]);
            }
            str.append("]");
            outFile.writeLine("cv_acc," + str);
        }

        StringBuilder str = new StringBuilder("[" + trainingClassificationResults.predictions[0]);
        for (int i = 1; i < trainingClassificationResults.predictions.length; i++) {
            str.append(":").append(trainingClassificationResults.predictions[i]);
        }
        str.append("]");
        outFile.writeLine("train_predictions," + str);


        outFile.closeFile();
    }

    public static boolean isDatasetDone(String outputPath) {
        File f1 = new File(outputPath + defaultSaveFilename);
        return f1.exists() && !f1.isDirectory();
    }

    public static void printSummary(String moduleName) {
        System.out.println("[" + moduleName + "] DatasetPath: " + Application.datasetPath);
        System.out.println("[" + moduleName + "] Problem: " + Application.problem);
        System.out.println("[" + moduleName + "] Classifier: " + Application.classifierName);
        System.out.println("[" + moduleName + "] ParamId: " + Application.paramId);
        System.out.println("[" + moduleName + "] ZNorm: " + Application.znorm);
        System.out.println();
    }
}
