using System;
using System.IO;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Diagnostics;
using System.Threading;
using MathNet.Numerics;


// Paper: "Visualizing Data using t-SNE" (2008), 
//   L. van der Maaten and G. Hinton.
// Python code: https://lvdmaaten.github.io/tsne/

namespace TSNE
{
    internal class TSNEProgram
    {
        public static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            Control.UseNativeMKL();
            Console.WriteLine("Active Linear Algebra Provider: " + Control.Describe());
            Console.WriteLine("\nBegin t-SNE with C# demo ");

            Console.WriteLine("\nLoading source data ");
            // load data from file
            //string ifn = "penguin_12.txt";
            //var X = TSNE.MatLoad(ifn, new int[] { 1, 2, 3, 4 }, ',', "#");
            //int maxIter = 500;
            //int perplexity = 3;

            //string ifn = "penguin_12.txt";
            //var X = TSNE.MatLoad(ifn, new int[] { 1, 2, 3, 4 }, ',', "#");
            //int maxIter = 300;
            //int perplexity = 3;

            //alternatively use the large MNIST test set and load2 for loading all columns without specifying them
            string ifn = "mnist_test.csv";
            var X = TSNE.MatLoad2(ifn, ',', "#");
            int maxIter = 500;
            int perplexity = 10;

            Console.WriteLine("Data loaded from " + ifn);

            //show first 10 lines of X to console
            Console.WriteLine("First 10 rows of data:");
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < X.ColumnCount; j++)
                {
                    Console.Write(X[i, j].ToString("F1").PadLeft(8));
                }
                Console.WriteLine("");
            }

            var BHresult = TSNEAccord.Reduce(X, 2, perplexity);

            var BTmatrix = TSNE.ToMatrix(BHresult);
            TSNE.MatShow(BTmatrix, 2, 10, 20);
            string ofn2 = "data_reducedAccord.txt";
            TSNE.MatSave(BTmatrix, ofn2, ',', 2);

            var umap = UMAPAnalysis.Reduce(X);
            var umapmatrix = TSNE.ToMatrix(umap);
            TSNE.MatShow(umapmatrix, 2, 10, 20);
            string ofn3 = "data_reducedUMAP.txt";
            TSNE.MatSave(umapmatrix, ofn3, ',', 2);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.WriteLine("\nApplying PCA reduction to 50 components ");
            // Step 1: Reduce dimensions with PCA (e.g., to 50 components)
            if (X.ColumnCount > 100)
            {
                // Perform PCA (Principal Component Analysis) on X 
                var Xpca = PCA.Reduce(X, 50);
                X = Xpca;
            }
           
            Console.WriteLine("\nApplying t-SNE reduction ");
            Console.WriteLine("Setting maxIter = " + maxIter);
            Console.WriteLine("Setting perplexity = " + perplexity);
            //Step 2: Apply t-SNE to reduce to 2 or 3 dimensions
            var reduced = TSNE.Reduce(X, maxIter, perplexity);
            sw.Stop();
            
            Console.WriteLine("\nReduced data: ");
            TSNE.MatShow(reduced, 2, 10, 20);

            Console.WriteLine("\nSaving reduced data for a graph ");
            string ofn = "data_reduced.txt";
            TSNE.MatSave(reduced, ofn, ',', 2);
            
            

            Console.WriteLine("t-SNE reduction completed in " + sw.ElapsedMilliseconds / 1000.0 + " s");
            Console.WriteLine("\nEnd t-SNE demo ");
            Console.ReadLine();
        }
    } // Program

    // ========================================================

} // end namespace TSNE

