using System;
using UMAP;

namespace TSNE
{
    public class UMAPAnalysis
    {
        /// <summary>
        /// Runs PCO (Principal Coordinates Analysis) as a prefilter, then t-SNE using Accord.NET.
        /// </summary>
        /// <param name="X">Input data as double[,] (rows: samples, cols: features)</param>
        /// <returns>Projected data as double[,]</returns>
        public static double[][] Reduce(MathNet.Numerics.LinearAlgebra.Matrix<double> X)
        {
            var umap = new Umap(); // sometimes libraries name type Umap or Umap_Umap depending on package; adjust to what your package exposes
                                   // Note: exact class name from the NuGet package may be "Umap" or similar - check Intellisense after installing.
                                   // The README uses: var umap = new Umap();
            var doublevectors = TSNE.ToDoubleArray(X); // convert to float[][]

                        
            // convert to float[][]
            float[][] vectors = new float[doublevectors.Length][];
            for (int i = 0; i < doublevectors.Length; i++)
            {
                vectors[i] = Array.ConvertAll(doublevectors[i], item => (float)item);
            }
            // === 3) Initialize and run training epochs ===
            int recommendedEpochs = umap.InitializeFit(vectors); // returns recommended number of epochs
            Console.WriteLine($"Recommended epochs: {recommendedEpochs}");

            for (int i = 0; i < recommendedEpochs; i++)
            {
                umap.Step();
                if ((i + 1) % 50 == 0) Console.WriteLine($"Epoch {i + 1}/{recommendedEpochs}");
            }

            // === 4) Get 2D embedding ===
            var embedding = umap.GetEmbedding(); // float[][] with embedding.Length == vectors.Length, each nested array length==2

            // Convert float[][] embedding to double[][]
            double[][] embeddingDouble = new double[embedding.Length][];
            for (int i = 0; i < embedding.Length; i++)
            {
                embeddingDouble[i] = Array.ConvertAll(embedding[i], item => (double)item);
            }

            return embeddingDouble;
        }
    }
}