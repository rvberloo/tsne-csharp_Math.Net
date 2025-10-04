using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;


// Paper: "Visualizing Data using t-SNE" (2008), 
//   L. van der Maaten and G. Hinton.
// Python code: https://lvdmaaten.github.io/tsne/

namespace TSNE
{
  internal class TSNEProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin t-SNE with C# demo ");

      Console.WriteLine("\nLoading source data ");
      // hard-coded demo data
      //double[][] X = new double[12][];  // penguin data
      //X[0] = new double[] { 39.5, 17.4, 186, 3800 };  // class 0
      //X[1] = new double[] { 40.3, 18.0, 195, 3250 };  // 0
      //X[2] = new double[] { 36.7, 19.3, 193, 3450 };  // 0
      //X[3] = new double[] { 38.9, 17.8, 181, 3625 };  // 0
      //X[4] = new double[] { 46.5, 17.9, 192, 3500 };  // 1
      //X[5] = new double[] { 45.4, 18.7, 188, 3525 };  // 1
      //X[6] = new double[] { 45.2, 17.8, 198, 3950 };  // 1
      //X[7] = new double[] { 46.1, 18.2, 178, 3250 };  // 1
      //X[8] = new double[] { 46.1, 13.2, 211, 4500 };  // 2
      //X[9] = new double[] { 48.7, 14.1, 210, 4450 };  // 2
      //X[10] = new double[] { 46.5, 13.5, 210, 4550 };  // 2
      //X[11] = new double[] { 45.4, 14.6, 211, 4800 };  // 2

      // load data from file
      string ifn = @"C:\VSM\TSNE\Data\penguin_12.txt";
      // # species, bill len, wid, flip len , weight
      // 0, 39.5, 17.4, 186, 3800
      // 0, 40.3, 18.0, 195, 3250
      // . . .
      // 2, 45.4, 14.6, 211, 4800
      double[][] X = TSNE.MatLoad(ifn,
        new int[] { 1, 2, 3, 4 }, ',', "#"); // not col [0]

      Console.WriteLine("\nSource data: ");
      TSNE.MatShow(X, 1, 10, true);

      Console.WriteLine("\nApplying t-SNE reduction ");
      int maxIter = 500;
      int perplexity = 3;
      Console.WriteLine("Setting maxIter = " + maxIter);
      Console.WriteLine("Setting perplaxity = " + perplexity);
      double[][] reduced = TSNE.Reduce(X, maxIter, perplexity);

      Console.WriteLine("\nReduced data: ");
      TSNE.MatShow(reduced, 2, 10, true);

      Console.WriteLine("\nSaving reduced data for a graph ");
      string ofn = @"C:\VSM\TSNE\Data\penguin_reduced.txt";
      TSNE.MatSave(reduced, ofn, ',', 2);

      Console.WriteLine("\nEnd t-SNE demo ");
      Console.ReadLine();
    }
  } // Program

  // ========================================================

  public class TSNE
  {
    // wrapper class for static Reduce() and its helpers

    public static double[][] Reduce(double[][] X, int maxIter, int perplexity)
    {
      int n = X.Length;
      double initialMomentum = 0.5;
      double finalMomentum = 0.8;
      double eta = 500.0;
      double minGain = 0.01;

      // initialize result Y using Math.NET
      Gaussian g = new Gaussian(mean: 0.0, sd: 1.0, seed: 1);
      var Y = Matrix<double>.Build.Dense(n, 2, (i, j) => g.NextGaussian());
      var dY = Matrix<double>.Build.Dense(n, 2);
      var iY = Matrix<double>.Build.Dense(n, 2);
      var Gains = Matrix<double>.Build.Dense(n, 2, 1.0);

      // Convert X to Math.NET matrix for ComputeP
      var P = ToMatrix(ComputeP(X, perplexity));
      P = MatAdd(P, MatTranspose(P));
      double sumP = MatSum(P);
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < n; ++j)
        {
          P[i, j] = (P[i, j] * 4.0) / sumP;
          if (P[i, j] < 1.0e-12)
            P[i, j] = 1.0e-12;
        }
      }

      for (int iter = 0; iter < maxIter; ++iter)
      {
        // rowSums = sum of squares of each row in Y
        var rowSums = Y.PointwisePower(2).RowSums();

        // Num = -2 * (Y * Y^T) and normalize
        var Num = MatProduct(Y, MatTranspose(Y)).Multiply(-2.0);
        for (int i = 0; i < n; ++i)
          Num.SetRow(i, Num.Row(i) + rowSums[i]);
        Num = Num.Transpose();
        for (int i = 0; i < n; ++i)
          Num.SetRow(i, Num.Row(i) + rowSums[i]);
        Num = Num.Add(1.0).PointwisePower(-1.0);
        for (int i = 0; i < n; ++i)
          Num[i, i] = 0.0;

        double sumNum = Num.Sum();
        var Q = Num.Clone();
        Q = Q.Divide(sumNum);
        Q.MapInplace(x => x < 1.0e-12 ? 1.0e-12 : x);

        var PminusQ = P - Q;

        // Compute dY using Math.NET Numerics
        for (int i = 0; i < n; ++i)
        {
          var tmpA = Y.Row(i);
          var tmpB = Matrix<double>.Build.Dense(n, 2, (r, c) => tmpA[c] - Y[r, c]);
          var tmpK = PminusQ.Column(i).PointwiseMultiply(Num.Column(i));
          var tmpF = Matrix<double>.Build.Dense(n, 2, (r, c) => tmpK[r]);
          var tmpG = tmpF.PointwiseMultiply(tmpB);
          var tmpZ = tmpG.ColumnSums();
          dY.SetRow(i, tmpZ);
        }

        double momentum = (iter < 20) ? initialMomentum : finalMomentum;

        // compute Gains using Math.NET
        for (int i = 0; i < n; ++i)
        {
          for (int j = 0; j < 2; ++j)
          {
            if ((dY[i, j] > 0.0 && iY[i, j] <= 0.0) || (dY[i, j] <= 0.0 && iY[i, j] > 0.0))
              Gains[i, j] = Gains[i, j] + 0.2;
            else if ((dY[i, j] > 0.0 && iY[i, j] > 0.0) || (dY[i, j] <= 0.0 && iY[i, j] <= 0.0))
              Gains[i, j] = Gains[i, j] * 0.8;
          }
        }
        Gains.MapInplace(x => x < minGain ? minGain : x);

        // use dY to compute iY to update result Y
        iY = iY.Multiply(momentum).Subtract(Gains.PointwiseMultiply(dY).Multiply(eta));
        Y = Y.Add(iY);

        // Center Y by subtracting column means
        var meansY = Y.ColumnSums() / n;
        var meansTile = Matrix<double>.Build.Dense(n, 2, (i, j) => meansY[j]);
        Y = Y - meansTile;

        if ((iter + 1) % 100 == 0)
        {
          double C = MatSum(MatMultLogDivide(P, P, Q));
          Console.WriteLine("iter = " + (iter + 1).ToString().PadLeft(6) + "  |  error = " + C.ToString("F4"));
        }

        if (iter == 100)
        {
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
              P[i, j] /= 4.0;
        }
      }
      return Y.ToRowArrays();
    }

    // ------------------------------------------------------

    private static double[][] ComputeP(double[][] X,
      int perplexity)
    {
      // helper for Reduce()
      double tol = 1.0e-5;
      int n = X.Length; int d = X[0].Length;
      double[][] D = MatSquaredDistances(X);
      double[][] P = MatCreate(n, n);
      double[] beta = new double[n];
      for (int i = 0; i < n; ++i)
        beta[i] = 1.0;
      double logU = Math.Log(perplexity);
      for (int i = 0; i < n; ++i)
      {
        double betaMin = -1.0e15;
        double betaMax = 1.0e15;
        // ith row of D[][], without 0.0 at [i][i]
        double[] Di = new double[n - 1];
        int k = 0; // points into Di
        for (int j = 0; j < n; ++j) // j points D[][]
        {
          if (j == i) continue;
          Di[k] = D[i][j];
          ++k;
        }

        double h;
        double[] currP = ComputePH(Di, beta[i], out h);
        double hDiff = h - logU;
        int ct = 0;
        while (Math.Abs(hDiff) > tol && ct < 50)
        {
          if (hDiff > 0.0)
          {
            betaMin = beta[i];
            if (betaMax == 1.0e15 || betaMax == -1.0e15)
              beta[i] *= 2.0;
            else
              beta[i] = (beta[i] + betaMax) / 2.0;
          }
          else
          {
            betaMax = beta[i];
            if (betaMin == 1.0e15 || betaMin == -1.0e15)
              beta[i] /= 2.0;
            else
              beta[i] = (beta[i] + betaMin) / 2.0;
          }
          // recompute
          currP = ComputePH(Di, beta[i], out h);
          hDiff = h - logU;
          ++ct;
        } // while

        k = 0; // points into currP vector
        for (int j = 0; j < n; ++j)
        {
          if (i == j) P[i][j] = 0.0;
          else P[i][j] = currP[k++];
        }
      } // for i

      return P;
    } // ComputeP

    // ------------------------------------------------------

    private static double[] ComputePH(double[] di,
      double beta, out double h)
    {
      // helper for ComputeP()
      // return p vector explicitly, h val as out param
      int n = di.Length; // (n-1) relative to D
      double[] p = new double[n];
      double sumP = 0.0;
      for (int j = 0; j < n; ++j)
      {
        p[j] = Math.Exp(-1 * di[j] * beta);
        sumP += p[j];
      }
      if (sumP == 0.0) sumP = 1.0e-12; // avoid div by 0

      double sumDP = 0.0;
      for (int j = 0; j < n; ++j)
        sumDP += di[j] * p[j];

      double hh = Math.Log(sumP) + (beta * sumDP / sumP);
      for (int j = 0; j < n; ++j)
        p[j] /= sumP;

      h = hh;
      return p; // a new row for P[][]
    }

    // ------------------------------------------------------
    //
    // 32 secondary helper functions, one helper class
    //
    // MatSquaredDistances, MatCreate, MatOnes,
    // MatLoad, MatSave, VecLoad,
    // MatShow, MatShow, MatShow, (3 overloads)
    // VecShow, VecShow, MatCopy, MatAdd, MatTranspose,
    // MatProduct, MatDiff, MatExtractRow,
    // MatExtractColumn, VecMinusMat, VecMult, VecTile,
    // MatMultiply, MatMultLogDivide, MatRowSums,
    // MatColumnSums, MatColumnMeans, MatReplaceRow,
    // MatVecAdd, MatAddScalar, MatInvertElements,
    // MatSum, MatZeroOutDiag
    //
    // Gaussian class NextGaussian()
    //
    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static double[][] MatSquaredDistances(double[][] X)
    {
      var mat = Matrix<double>.Build.DenseOfRows(X);
      int n = mat.RowCount;
      var result = Matrix<double>.Build.Dense(n, n);
      for (int i = 0; i < n; ++i)
      {
        var rowI = mat.Row(i);
        for (int j = i; j < n; ++j)
        {
          var rowJ = mat.Row(j);
          double dist = (rowI - rowJ).PointwisePower(2).Sum();
          result[i, j] = dist;
          result[j, i] = dist;
        }
      }
      return result.ToRowArrays();
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatCreate(int rows, int cols)
    {
      return Matrix<double>.Build.Dense(rows, cols);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatOnes(int rows, int cols)
    {
      return Matrix<double>.Build.Dense(rows, cols, 1.0);
    }

    // ------------------------------------------------------

    public static double[][] MatLoad(string fn,
      int[] usecols, char sep, string comment)
    {
      // count number of non-comment lines
      int nRows = 0;
      string line = "";
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
        if (line.StartsWith(comment) == false)
          ++nRows;
      sr.Close(); ifs.Close();

      // make result matrix
      int nCols = usecols.Length;
      double[][] result = new double[nRows][];
      for (int r = 0; r < nRows; ++r)
        result[r] = new double[nCols];

      line = "";
      string[] tokens = null;
      ifs = new FileStream(fn, FileMode.Open);
      sr = new StreamReader(ifs);

      int i = 0;
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        tokens = line.Split(sep);
        for (int j = 0; j < nCols; ++j)
        {
          int k = usecols[j];  // into tokens
          result[i][j] = double.Parse(tokens[k]);
        }
        ++i;
      }
      sr.Close(); ifs.Close();
      return result;
    }

    // ------------------------------------------------------

    public static void MatSave(double[][] data,
      string fn, char sep, int dec)
    {
      int nRows = data.Length;
      int nCols = data[0].Length;
      FileStream ofs = new FileStream(fn,
        FileMode.Create);
      StreamWriter sw = new StreamWriter(ofs);
      for (int i = 0; i < nRows; ++i)
      {
        string line = "";
        for (int j = 0; j < nCols; ++j)
        {
          line += data[i][j].ToString("F" + dec);
          if (j < nCols - 1)
            line += sep;
        }
        sw.WriteLine(line); // includes NL
      }

      sw.Close();
      ofs.Close();
    }

    // ------------------------------------------------------

    public static int[] VecLoad(string fn, int usecol,
      string comment)
    {
      char dummySep = ',';
      double[][] tmp = MatLoad(fn, new int[] { usecol },
        dummySep, comment);
      int n = tmp.Length;
      int[] result = new int[n];
      for (int i = 0; i < n; ++i)
        result[i] = (int)tmp[i][0];
      return result;
    }

    // ------------------------------------------------------

    public static void MatShow(double[][] M, int dec,
      int wid, bool showIndices)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < M.Length; ++i)
      {
        if (showIndices == true)
        {
          int pad = M.Length.ToString().Length;
          Console.Write("[" + i.ToString().
            PadLeft(pad) + "]");
        }
        for (int j = 0; j < M[0].Length; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine("");
      }
    }

    // ------------------------------------------------------

    public static void MatShow(double[][] M, int nCols,
      int dec, int wid, bool showIndices)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < M.Length; ++i)
      {
        if (showIndices == true)
        {
          int pad = M.Length.ToString().Length;
          Console.Write("[" + i.ToString().
            PadLeft(pad) + "]");
        }
        for (int j = 0; j < nCols; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine(" . . . ");
      }
    }

    // ------------------------------------------------------

    public static void MatShow(double[][] M, int dec,
      int wid, int nRows)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < nRows; ++i)
      {
        for (int j = 0; j < M[0].Length; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine("");
      }
      if (nRows < M.Length)
        Console.WriteLine(". . . ");
    }

    // ------------------------------------------------------

    public static void VecShow(int[] vec, int wid)
    {
      int n = vec.Length;
      for (int i = 0; i < n; ++i)
        Console.Write(vec[i].ToString().PadLeft(wid));
      Console.WriteLine("");
    }

    // ------------------------------------------------------

    public static void VecShow(double[] vec, int decimals,
      int wid)
    {
      int n = vec.Length;
      for (int i = 0; i < n; ++i)
        Console.Write(vec[i].ToString("F" + decimals).
          PadLeft(wid));
      Console.WriteLine("");
    }

    // ------------------------------------------------------

    private static double[][] MatCopy(double[][] m)
    {
      int nRows = m.Length; int nCols = m[0].Length;
      double[][] result = MatCreate(nRows, nCols);
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[i][j] = m[i][j];
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatAdd(Matrix<double> mA, Matrix<double> mB)
    {
      return mA + mB;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatTranspose(Matrix<double> m)
    {
      return m.Transpose();
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatProduct(Matrix<double> matA, Matrix<double> matB)
    {
      return matA * matB;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatDiff(Matrix<double> A, Matrix<double> B)
    {
      return A - B;
    }

    // ------------------------------------------------------

    private static double[] MatExtractRow(double[][] M,
      int row)
    {
      int cols = M[0].Length;
      double[] result = new double[cols];
      for (int j = 0; j < cols; ++j)
        result[j] = M[row][j];
      return result;
    }

    // ------------------------------------------------------

    private static double[] MatExtractColumn(double[][] M,
      int col)
    {
      int rows = M.Length;
      double[] result = new double[rows];
      for (int i = 0; i < rows; ++i)
        result[i] = M[i][col];
      return result;
    }

    // ------------------------------------------------------

    private static double[][] VecMinusMat(double[] v,
      double[][] M)
    {
      // add v row vector to each row of -M
      // B = A - Y  # 10x2; -Y + A vector to each row
      double[][] result = MatCopy(M);
      for (int i = 0; i < M.Length; ++i)
        for (int j = 0; j < M[0].Length; ++j)
          result[i][j] = -M[i][j] + v[j];
      return result;
    }

    // ------------------------------------------------------

    private static double[] VecMult(double[] v1,
      double[] v2)
    {
      // element-wise multiplication
      int n = v1.Length;
      double[] result = new double[n];
      for (int i = 0; i < n; ++i)
        result[i] = v1[i] * v2[i];
      return result;
    }

    // ------------------------------------------------------

    private static double[][] VecTile(double[] vec,
      int n)
    {
      // n row-copies of vec
      int nCols = vec.Length;
      double[][] result = MatCreate(n, nCols);
      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < nCols; ++j)
        {
          result[i][j] = vec[j];
        }
      }
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics for elementwise multiplication
    private static Matrix<double> MatMultiply(Matrix<double> A, Matrix<double> B)
    {
      return A.PointwiseMultiply(B);
    }

    // ------------------------------------------------------

    private static double[][] MatMultLogDivide(double[][] P,
      double[][] A, double[][] B)
    {
      // element-wise P * log(A / B)
      int nRows = A.Length; int nCols = A[0].Length;
      double[][] result = MatCreate(nRows, nCols);
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
        {
          if (B[i][j] == 0.0)
            result[i][j] = P[i][j] * Math.Log(1.0e10);
          else
            result[i][j] = P[i][j] *
              Math.Log(A[i][j] / B[i][j]);
        }
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Vector<double> MatRowSums(Matrix<double> M)
    {
      return M.RowSums();
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Vector<double> MatColumnSums(Matrix<double> M)
    {
      return M.ColumnSums();
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Vector<double> MatColumnMeans(Matrix<double> M)
    {
      return M.ColumnSums() / M.RowCount;
    }

    // ------------------------------------------------------

    private static void MatReplaceRow(double[][] M, int i,
      double[] vec)
    {
      int nCols = M[0].Length;
      for (int j = 0; j < nCols; ++j)
        M[i][j] = vec[j];
      return;
    }

    // ------------------------------------------------------

    private static double[][] MatVecAdd(double[][] M,
      double[] vec)
    {
      // add row vector vec to each row of M
      int nr = M.Length; int nc = M[0].Length;
      double[][] result = MatCreate(nr, nc);
      for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[i][j] = M[i][j] + vec[j];
      return result;
    }

    // ------------------------------------------------------

    private static double[][] MatAddScalar(double[][] M,
      double val)
    {
      int nr = M.Length; int nc = M[0].Length;
      double[][] result = MatCreate(nr, nc);
      for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[i][j] = M[i][j] + val;
      return result;
    }

    // ------------------------------------------------------

    private static double[][] MatInvertElements(double[][] M)
    {
      int nr = M.Length; int nc = M[0].Length;
      double[][] result = MatCreate(nr, nc);
      for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[i][j] = 1.0 / M[i][j]; // M[i][j] not 0
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static double MatSum(Matrix<double> M)
    {
      return M.Sum();
    }

    // ------------------------------------------------------

    static void MatZeroOutDiag(double[][] M)
    {
      int nr = M.Length; //int nc = M[0].Length;
      double result = 0.0;
      for (int i = 0; i < nr; ++i)
        M[i][i] = 0.0;
      return;
    }

    // ------------------------------------------------------

    // nested Gaussian to init TSNE.Reduce() result
    class Gaussian
    {
      private Random rnd;
      private double mean;
      private double sd;

      public Gaussian(double mean, double sd, int seed)
      {
        this.rnd = new Random(seed);
        this.mean = mean;
        this.sd = sd;
      }

      public double NextGaussian()
      {
        double u1 = this.rnd.NextDouble();
        double u2 = this.rnd.NextDouble();
        double left = Math.Cos(2.0 * Math.PI * u1);
        double right = Math.Sqrt(-2.0 * Math.Log(u2));
        double z = left * right;
        return this.mean + (z * this.sd);
      }
    } // Gaussian

    // ------------------------------------------------------

    // Conversion helpers between double[][] and Matrix<double>
    private static Matrix<double> ToMatrix(double[][] array)
    {
      return Matrix<double>.Build.DenseOfRows(array);
    }

    private static double[][] ToArray(Matrix<double> matrix)
    {
      return matrix.ToRowArrays();
    }

  } // class TSNE

  // ========================================================

} // ns
