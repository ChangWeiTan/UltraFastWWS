package utils;

public class EfficientSymmetricMatrix {
    int matrixSize;
    double[][] distanceVector;

    public EfficientSymmetricMatrix(int N) {
        distanceVector = new double[1][N * (N + 1) / 2];
        matrixSize = N;
    }

    public EfficientSymmetricMatrix(int N, int nParams) {
        distanceVector = new double[nParams][N * (N + 1) / 2];
        matrixSize = N;
    }

    public int indexOf(int i, int j) {
        if (i <= j)
            return i * matrixSize - (i - 1) * i / 2 + j - i;
        else
            return j * matrixSize - (j - 1) * j / 2 + i - j;
    }

    public double get(int i, int j) {
        int idx = indexOf(i, j);
        return distanceVector[0][idx];
    }

    public double get(int i, int j, int paramId) {
        int idx = indexOf(i, j);
        return distanceVector[paramId][idx];
    }

    public void put(int i, int j, double val) {
        int idx = indexOf(i, j);
        distanceVector[0][idx] = val;
    }

    public void put(int i, int j, double val, int paramId) {
        int idx = indexOf(i, j);
        distanceVector[paramId][idx] = val;
    }
}
