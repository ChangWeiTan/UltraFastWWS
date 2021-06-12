package utils;

import java.util.*;

public class GenericTools {
    static Random random = new Random(100);

    public static String doTime(double elapsedTimeNanoSeconds) {
        final double duration = elapsedTimeNanoSeconds / 1e6;
        return String.format("%d s %.3f ms", (int) (duration / 1000), (duration % 1000));
    }

    public static String doTimeNs(double elapsedTimeNanoSeconds) {
        int hour = (int) (elapsedTimeNanoSeconds / 3.6e+12);
        int min = (int) (elapsedTimeNanoSeconds / 6e+10);
        int s = (int) (elapsedTimeNanoSeconds / 1e9);
        int ms = (int) (elapsedTimeNanoSeconds / 1e6);
        int us = (int) (elapsedTimeNanoSeconds / 1e3);
        StringBuilder str = new StringBuilder();
        if (hour > 0)
            str.append((hour % 60)).append(" H ");
        if (min > 0)
            str.append((min % 60)).append(" M ");
        if (s > 0)
            str.append((s % 60)).append(" s ");
        if (ms > 0)
            str.append((ms % 1000)).append(" ms ");
        if (us > 0)
            str.append((us % 1000)).append(" us ");

        str.append(((int) (elapsedTimeNanoSeconds % 1000))).append(" ns");

        return str.toString();
    }

    public static boolean isMissing(double a) {
        return Double.isNaN(a);
    }

    public static double[] fillWithNoise(final double[] data, final int maxLen) {
        final int seqLen = data.length;
        final double[] arr = new double[maxLen];

        System.arraycopy(data, 0, arr, 0, seqLen);

        for (int i = 0; i < maxLen; i++) {
            if (isMissing(arr[i]))
                arr[i] = random.nextDouble() / 1000;
        }
        return arr;
    }

    public static double[] znormalise(final double[] values) {
        double sum = 0.0;
        double standardDeviation = 0.0;
        int length = values.length;

        for (double num : values) {
            sum += num;
        }
        double m = sum / length;

        for (double num : values) {
            standardDeviation += Math.pow(num - m, 2);
        }
        double sd = Math.sqrt(standardDeviation / (length));

        final double[] normalizedValues = new double[length];
        if (sd <= 0)
            sd = 1;
        for (int i = 0; i < length; i++) {
            if (Double.isNaN(values[i])) {
                normalizedValues[i] = values[i];
            } else {
                normalizedValues[i] = (values[i] - m) / sd;
            }
        }
        return normalizedValues;
    }

    public static double min3(final double a, final double b, final double c) {
        return (a <= b) ? (Math.min(a, c)) : Math.min(b, c);
    }

    public static int argMin3(final double a, final double b, final double c) {
        return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
    }

    public static void println(Object str) {
        System.out.println(str);
    }

}
