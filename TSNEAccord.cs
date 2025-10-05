using System;
using Accord.Math;
using Accord.Statistics.Analysis;
using Accord.MachineLearning.Clustering; // <-- Add this import

namespace TSNE
{
    public class TSNEAccord
    {
        /// <summary>
        /// Runs PCO (Principal Coordinates Analysis) as a prefilter, then t-SNE using Accord.NET.
        /// </summary>
        /// <param name="X">Input data as double[,] (rows: samples, cols: features)</param>
        /// <param name="outputDims">Number of output dimensions (usually 2)</param>
        /// <param name="perplexity">t-SNE perplexity</param>
        /// <param name="maxIter">t-SNE max iterations</param>
        /// <returns>Projected data as double[,]</returns>
        public static double[][] Reduce(MathNet.Numerics.LinearAlgebra.Matrix<double> X, int outputDims = 2, double perplexity = 30)
        {
            if (X.ColumnCount > 100)
            {
                // Perform PCA (Principal Component Analysis) on X
                var Xpca = PCA.Reduce(X, 50);
                X = Xpca;
            }

            // 2. t-SNE using Accord.NET's TSNE in Accord.MachineLearning.Clustering
            
            var tsne = new Accord.MachineLearning.Clustering.TSNE
            {
                NumberOfOutputs = outputDims,
                Perplexity = perplexity
            };
            var Xmat = TSNE.ToDoubleArray(X);

            double[][] output = tsne.Transform(Xmat);

            // Make it 1-dimensional
            double[] y = output.Reshape();

            return output;
        }
    }
}