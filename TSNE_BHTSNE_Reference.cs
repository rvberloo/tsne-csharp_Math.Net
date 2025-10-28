using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace TSNE
{
    // Direct translation of Barnes-Hut t-SNE reference logic from bhtsne.cpp
    public class TSNE_BHTSNE_Reference
    {
        // Main t-SNE loop (simplified, not optimized)
        public static Matrix<double> Reduce(Matrix<double> X, int maxIter, int perplexity, double theta = 0.5)
        {
            int n = X.RowCount;
            int d = X.ColumnCount;
            int no_dims = 2;
            double initialMomentum = 0.5;
            double finalMomentum = 0.8;
            double eta = 200.0;
            double minGain = 0.01;

            // Initialize Y
            var rand = new Random(1);
            var Y = Matrix<double>.Build.Dense(n, no_dims);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < no_dims; ++j)
                    Y[i, j] = NextGaussian(rand) * 0.0001;

            var dY = Matrix<double>.Build.Dense(n, no_dims);
            var uY = Matrix<double>.Build.Dense(n, no_dims);
            var Gains = Matrix<double>.Build.Dense(n, no_dims, 1.0);

            // Compute P (using same logic as reference)
            var P = ComputeP(X, perplexity);
            P = P + P.Transpose();
            double sumP = P.Enumerate().Sum();
            P = P.Multiply(1.0 / sumP);
            P.MapInplace(x => x < 1e-12 ? 1e-12 : x);
            P = P.Multiply(12.0); // Lie about P for first iterations

            for (int iter = 0; iter < maxIter; ++iter)
            {
                Console.WriteLine($"Iteration: {iter}");
                // Build Barnes-Hut tree
                double[,] Yarr = new double[n, no_dims];
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < no_dims; ++j)
                        Yarr[i, j] = Y[i, j];
                double minX = Y.Column(0).Minimum();
                double maxX = Y.Column(0).Maximum();
                double minY = Y.Column(1).Minimum();
                double maxY = Y.Column(1).Maximum();
                var indices = new List<int>();
                for (int i = 0; i < n; ++i) indices.Add(i);
                var tree = new Quadtree(Yarr, indices, minX, minY, maxX, maxY, 1);

                // Attractive forces (sparse P)
                var pos_f = Matrix<double>.Build.Dense(n, no_dims);
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        if (P[i, j] > 0)
                        {
                            double dx = Y[i, 0] - Y[j, 0];
                            double dy = Y[i, 1] - Y[j, 1];
                            pos_f[i, 0] += P[i, j] * dx;
                            pos_f[i, 1] += P[i, j] * dy;
                        }
                    }
                }

                // Repulsive forces (Barnes-Hut)
                var neg_f = Matrix<double>.Build.Dense(n, no_dims);
                double sum_Q = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    double fx = 0.0, fy = 0.0, sq = 0.0;
                    tree.ComputeRepulsiveForce(Yarr, Y[i, 0], Y[i, 1], theta, ref fx, ref fy, ref sq, i);
                    neg_f[i, 0] = fx;
                    neg_f[i, 1] = fy;
                    sum_Q += sq;
                }

                // Gradient update
                for (int i = 0; i < n; ++i)
                {
                    dY[i, 0] = pos_f[i, 0] - (neg_f[i, 0] / sum_Q);
                    dY[i, 1] = pos_f[i, 1] - (neg_f[i, 1] / sum_Q);
                }

                // Update gains
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < no_dims; ++j)
                    {
                        if ((dY[i, j] > 0.0 && uY[i, j] <= 0.0) || (dY[i, j] <= 0.0 && uY[i, j] > 0.0))
                            Gains[i, j] = Gains[i, j] + 0.2;
                        else
                            Gains[i, j] = Gains[i, j] * 0.8;
                        if (Gains[i, j] < minGain) Gains[i, j] = minGain;
                    }
                }

                // Update Y
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < no_dims; ++j)
                        uY[i, j] = initialMomentum * uY[i, j] - eta * Gains[i, j] * dY[i, j];
                Y = Y + uY;

                // Zero-mean
                var meansY = Y.ColumnSums() / n;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < no_dims; ++j)
                        Y[i, j] -= meansY[j];

                // Stop lying about P after 100 iterations
                if (iter == 100)
                    P.MapInplace(x => x / 12.0);
            }
            return Y;
        }

        // Helper: Gaussian random
        private static double NextGaussian(Random rnd)
        {
            double u1 = rnd.NextDouble();
            double u2 = rnd.NextDouble();
            double left = Math.Cos(2.0 * Math.PI * u1);
            double right = Math.Sqrt(-2.0 * Math.Log(u2));
            return left * right;
        }

        // Helper: Compute P (dense, not optimized)
        private static Matrix<double> ComputeP(Matrix<double> X, int perplexity)
        {
            int n = X.RowCount;
            var D = Matrix<double>.Build.Dense(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    D[i, j] = (X.Row(i) - X.Row(j)).PointwisePower(2).Sum();
            var P = Matrix<double>.Build.Dense(n, n);
            double tol = 1e-5;
            for (int i = 0; i < n; ++i)
            {
                double beta = 1.0;
                double betaMin = double.NegativeInfinity;
                double betaMax = double.PositiveInfinity;
                double[] Di = new double[n - 1];
                int k = 0;
                for (int j = 0; j < n; ++j)
                {
                    if (j == i) continue;
                    Di[k++] = D[i, j];
                }
                double h, sumP;
                double[] currP = new double[n - 1];
                int ct = 0;
                while (true)
                {
                    for (int j = 0; j < n - 1; ++j)
                        currP[j] = Math.Exp(-beta * Di[j]);
                    sumP = 0.0;
                    for (int j = 0; j < n - 1; ++j)
                        sumP += currP[j];
                    if (sumP == 0.0) sumP = 1e-12;
                    h = 0.0;
                    for (int j = 0; j < n - 1; ++j)
                        h += beta * Di[j] * currP[j];
                    h = (h / sumP) + Math.Log(sumP);
                    double hDiff = h - Math.Log(perplexity);
                    if (Math.Abs(hDiff) < tol || ct > 50) break;
                    if (hDiff > 0.0)
                    {
                        betaMin = beta;
                        if (double.IsInfinity(betaMax))
                            beta *= 2.0;
                        else
                            beta = (beta + betaMax) / 2.0;
                    }
                    else
                    {
                        betaMax = beta;
                        if (double.IsInfinity(betaMin))
                            beta /= 2.0;
                        else
                            beta = (beta + betaMin) / 2.0;
                    }
                    ct++;
                }
                k = 0;
                for (int j = 0; j < n; ++j)
                {
                    if (i == j) P[i, j] = 0.0;
                    else P[i, j] = currP[k++] / sumP;
                }
            }
            return P;
        }
    }
}
