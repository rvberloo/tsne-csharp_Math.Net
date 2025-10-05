using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;

namespace TSNE
{
    public class PCA
    {
        /// <summary>
        /// Performs Principal Component Analysis (PCA) on the input matrix.
        /// Each row of X is an observation, each column is a variable.
        /// </summary>
        /// <param name="X">Input data matrix (n_samples x n_features)</param>
        /// <param name="nComponents">Number of principal components to keep</param>
        /// <returns>Transformed matrix with reduced dimensions (n_samples x nComponents)</returns>
        public static Matrix<double> Reduce(Matrix<double> X, int nComponents)
        {
            // Center the data (subtract mean of each column)
            var mean = X.ColumnSums() / X.RowCount;
            var Xcentered = X - DenseMatrix.Create(X.RowCount, X.ColumnCount, (i, j) => mean[j]);

            // Compute covariance matrix
            var cov = (Xcentered.TransposeThisAndMultiply(Xcentered)) / (X.RowCount - 1);

            // Eigen decomposition
            var evd = cov.Evd(Symmetricity.Symmetric);

            // Get eigenvalues and eigenvectors
            var eigenValues = evd.EigenValues.Real();
            var eigenVectors = evd.EigenVectors;

            // Sort eigenvalues and eigenvectors in descending order
            var idx = eigenValues.EnumerateIndexed()
                .OrderByDescending(x => x.Item2)
                .Select(x => x.Item1)
                .ToArray();

            var selectedVectors = DenseMatrix.OfColumns(
                eigenVectors.RowCount,
                nComponents,
                idx.Take(nComponents).Select(i => eigenVectors.Column(i))
            );

            // Project data onto principal components
            return Xcentered * selectedVectors;
        }
    }
}