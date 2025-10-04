using System;
using System.IO;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;


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
      string ifn = "penguin_12.txt";
      // # species, bill len, wid, flip len , weight
      // 0, 39.5, 17.4, 186, 3800
      // 0, 40.3, 18.0, 195, 3250
      // . . .
      // 2, 45.4, 14.6, 211, 4800

      var X = TSNE.MatLoad(ifn, new int[] { 1, 2, 3, 4 }, ',', "#");
      Console.WriteLine("\nSource data: ");
      TSNE.MatShow(X, 1, 10, true);

      Console.WriteLine("\nApplying t-SNE reduction ");
      int maxIter = 500;
      int perplexity = 3;
      Console.WriteLine("Setting maxIter = " + maxIter);
      Console.WriteLine("Setting perplaxity = " + perplexity);
      var reduced = TSNE.Reduce(X, maxIter, perplexity);

      Console.WriteLine("\nReduced data: ");
      TSNE.MatShow(reduced, 2, 10, true);

      Console.WriteLine("\nSaving reduced data for a graph ");
      string ofn = "penguin_reduced.txt";
      TSNE.MatSave(reduced, ofn, ',', 2);

      Console.WriteLine("\nEnd t-SNE demo ");
      Console.ReadLine();
    }
  } // Program

  // ========================================================

  public class TSNE

  {
    // Computes conditional probabilities and entropy for a distance vector and beta
    private static double[] ComputePH(double[] di, double beta, out double h)
    {
      var vdi = Vector<double>.Build.DenseOfArray(di);
      var p = (-vdi * beta).PointwiseExp();
      double sumP = p.Sum();
      if (sumP == 0.0) sumP = 1.0e-12; // avoid div by 0
      double sumDP = vdi.PointwiseMultiply(p).Sum();
      double hh = Math.Log(sumP) + (beta * sumDP / sumP);
      p = p / sumP;
      h = hh;
      return p.ToArray();
    }

    public static Matrix<double> Reduce(Matrix<double> X, int maxIter, int perplexity)
    {
      int n = X.RowCount;
      double initialMomentum = 0.5;
      double finalMomentum = 0.8;
      double eta = 500.0;
      double minGain = 0.01;

      Gaussian g = new Gaussian(mean: 0.0, sd: 1.0, seed: 1);
      var Y = Matrix<double>.Build.Dense(n, 2);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < 2; ++j)
          Y[i, j] = g.NextGaussian();

      //var tmpy = ToDoubleArray(Y); for debugging
      var dY = Matrix<double>.Build.Dense(n, 2);
      var iY = Matrix<double>.Build.Dense(n, 2);
      var Gains = Matrix<double>.Build.Dense(n, 2, 1.0);

      var P = ComputeP(X, perplexity);
      P = MatAdd(P, MatTranspose(P));
      double sumP = MatSum(P);
      P = P.Multiply(4.0 / sumP);
      P.MapInplace(x => x < 1.0e-12 ? 1.0e-12 : x);

      for (int iter = 0; iter < maxIter; ++iter)
      {
        var rowSums = Y.PointwisePower(2).RowSums();
        var Num = MatProduct(Y, MatTranspose(Y)).Multiply(-2.0);
        for (int i = 0; i < n; ++i)
          Num.SetRow(i, Num.Row(i) + rowSums[i]);
        Num = Num.Transpose();
        for (int i = 0; i < n; ++i)
          Num.SetRow(i, Num.Row(i) + rowSums[i]);
        Num = Num.Add(1.0).PointwisePower(-1.0);
        for (int i = 0; i < n; ++i)
          Num[i, i] = 0.0;

        double sumNum = MatSum(Num);
        var Q = Num.Clone();
        Q = Q.Divide(sumNum);
        Q.MapInplace(x => x < 1.0e-12 ? 1.0e-12 : x);

        var PminusQ = P - Q;

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

        iY = iY.Multiply(momentum).Subtract(Gains.PointwiseMultiply(dY).Multiply(eta));
        Y = Y.Add(iY);

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

      return Y;
    }

    // ------------------------------------------------------

    private static Matrix<double> ComputeP(Matrix<double> X, int perplexity)
    {
      // helper for Reduce()
      double tol = 1.0e-5;
      int n = X.RowCount; int d = X.ColumnCount;
      var D = MatSquaredDistances(X);
      var P = Matrix<double>.Build.Dense(n, n);
      double[] beta = new double[n];
      for (int i = 0; i < n; ++i)
      {
        beta[i] = 1.0; // Initialize beta to 1.0 for each row
        double betaMin = -1.0e15;
        double betaMax = 1.0e15;
        // ith row of D, without 0.0 at [i,i]
        double[] Di = new double[n - 1];
        int k = 0;
        for (int j = 0; j < n; ++j)
        {
          if (j == i) continue;
          Di[k] = D[i, j];
          ++k;
        }

        double h;
        double[] currP = ComputePH(Di, beta[i], out h);
        double hDiff = h - Math.Log(perplexity);
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
          currP = ComputePH(Di, beta[i], out h);
          hDiff = h - Math.Log(perplexity);
          ++ct;
        }

        k = 0;
        for (int j = 0; j < n; ++j)
        {
          if (i == j) P[i, j] = 0.0;
          else P[i, j] = currP[k++];
        }
      }
      //var temp = ToDoubleArray(P); // for debugging
      return P;
    }
    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatSquaredDistances(Matrix<double> X)
    {
      int n = X.RowCount;
      var result = Matrix<double>.Build.Dense(n, n);
      for (int i = 0; i < n; ++i)
      {
        var rowI = X.Row(i);
        for (int j = i; j < n; ++j)
        {
          var rowJ = X.Row(j);
          double dist = (rowI - rowJ).PointwisePower(2).Sum();
          result[i, j] = dist;
          result[j, i] = dist;
        }
      }
      return result;
    }

    // ------------------------------------------------------

    private static Matrix<double> MatCreate(int rows, int cols)
    {
      return Matrix<double>.Build.Dense(rows, cols);
    }

    // ------------------------------------------------------

    private static Matrix<double> MatOnes(int rows, int cols)
    {
      return Matrix<double>.Build.Dense(rows, cols, 1.0);
    }

    // ------------------------------------------------------

    public static Matrix<double> MatLoad(string fn, int[] usecols, char sep, string comment)
    {
      var rows = new System.Collections.Generic.List<double[]>();
      using (var sr = new StreamReader(fn))
      {
        string line;
        while ((line = sr.ReadLine()) != null)
        {
          if (line.StartsWith(comment)) continue;
          var tokens = line.Split(sep);
          var row = new double[usecols.Length];
          for (int j = 0; j < usecols.Length; ++j)
            row[j] = double.Parse(tokens[usecols[j]], System.Globalization.CultureInfo.InvariantCulture);
          rows.Add(row);
        }
      }
      return Matrix<double>.Build.DenseOfRows(rows);
    }

    // ------------------------------------------------------

    public static void MatSave(Matrix<double> data, string fn, char sep, int dec)
    {
      int nRows = data.RowCount;
      int nCols = data.ColumnCount;
      using (var ofs = new FileStream(fn, FileMode.Create))
      using (var sw = new StreamWriter(ofs))
      {
        for (int i = 0; i < nRows; ++i)
        {
          string line = "";
          for (int j = 0; j < nCols; ++j)
          {
            line += data[i, j].ToString("F" + dec);
            if (j < nCols - 1)
              line += sep;
          }
          sw.WriteLine(line);
        }
      }
    }

    // ------------------------------------------------------

    public static Vector<double> VecLoad(string fn, int usecol, string comment)
    {
      char dummySep = ',';
      var mat = MatLoad(fn, new int[] { usecol }, dummySep, comment);
      return mat.Column(0);
    }

    public static void MatShow(Matrix<double> M, int dec, int wid, bool showIndices)
    {
      double small = 1.0 / Math.Pow(10, dec);
      int nRows = M.RowCount;
      int nCols = M.ColumnCount;
      for (int i = 0; i < nRows; ++i)
      {
        if (showIndices)
        {
          int pad = nRows.ToString().Length;
          Console.Write("[" + i.ToString().PadLeft(pad) + "]");
        }
        for (int j = 0; j < nCols; ++j)
        {
          double v = M[i, j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).PadLeft(wid));
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

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatCopy(Matrix<double> m)
    {
      return m.Clone();
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

    // Refactored to use Math.NET Numerics
    private static Vector<double> MatExtractRow(Matrix<double> M, int row)
    {
      return M.Row(row);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Vector<double> MatExtractColumn(Matrix<double> M, int col)
    {
      return M.Column(col);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> VecMinusMat(Vector<double> v, Matrix<double> M)
    {
      // Returns a matrix where each row is v - M.Row(i)
      int nRows = M.RowCount;
      int nCols = M.ColumnCount;
      var result = Matrix<double>.Build.Dense(nRows, nCols);
      for (int i = 0; i < nRows; ++i)
      {
        result.SetRow(i, v - M.Row(i));
      }
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Vector<double> VecMult(Vector<double> v1, Vector<double> v2)
    {
      return v1.PointwiseMultiply(v2);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> VecTile(Vector<double> vec, int n)
    {
      int nCols = vec.Count;
      var result = Matrix<double>.Build.Dense(n, nCols, (i, j) => vec[j]);
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics for elementwise multiplication
    private static Matrix<double> MatMultiply(Matrix<double> A, Matrix<double> B)
    {
      return A.PointwiseMultiply(B);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatMultLogDivide(Matrix<double> P, Matrix<double> A, Matrix<double> B)
    {
      // element-wise P * log(A / B), with safe log for B==0
      int nRows = A.RowCount;
      int nCols = A.ColumnCount;
      var result = Matrix<double>.Build.Dense(nRows, nCols);
      for (int i = 0; i < nRows; ++i)
      {
        for (int j = 0; j < nCols; ++j)
        {
          double b = B[i, j];
          double a = A[i, j];
          double p = P[i, j];
          if (b == 0.0)
            result[i, j] = p * Math.Log(1.0e10);
          else
            result[i, j] = p * Math.Log(a / b);
        }
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

    // Refactored to use Math.NET Numerics
    private static void MatReplaceRow(Matrix<double> M, int i, Vector<double> vec)
    {
      M.SetRow(i, vec);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatVecAdd(Matrix<double> M, Vector<double> vec)
    {
      // add row vector vec to each row of M
      int nr = M.RowCount;
      int nc = M.ColumnCount;
      var result = Matrix<double>.Build.Dense(nr, nc);
      for (int i = 0; i < nr; ++i)
      {
        result.SetRow(i, M.Row(i) + vec);
      }
      return result;
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatAddScalar(Matrix<double> M, double val)
    {
      return M.Add(val);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static Matrix<double> MatInvertElements(Matrix<double> M)
    {
      return M.Map(x => 1.0 / x);
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static double MatSum(Matrix<double> M)
    {
      return M.Enumerate().Sum();
    }

    // ------------------------------------------------------

    // Refactored to use Math.NET Numerics
    private static void MatZeroOutDiag(Matrix<double> M)
    {
      int n = Math.Min(M.RowCount, M.ColumnCount);
      for (int i = 0; i < n; ++i)
        M[i, i] = 0.0;
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

    private static double[][] ToDoubleArray(Matrix<double> mat)
    {
      int nRows = mat.RowCount;
      int nCols = mat.ColumnCount;
      double[][] array = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
      {
        array[i] = new double[nCols];
        for (int j = 0; j < nCols; ++j)
          array[i][j] = mat[i, j];
      }
      return array;
    }

  } // end class TSNE
} // end namespace TSNE

