using System;
using System.Collections.Generic;

namespace TSNE
{
    // Minimal Quadtree for 2D Barnes-Hut t-SNE
    public class Quadtree
    {
        public double MinX, MinY, MaxX, MaxY;
        public List<int> Points; // indices of points in this node
        public Quadtree?[]? Children { get; private set; } // 4 children, nullable
        public bool IsLeaf => Children == null;
        public double CenterX, CenterY; // center of mass
        public int Count; // number of points

        // Barnes-Hut repulsive force calculation for t-SNE
        // Returns the repulsive force vector [fx, fy] for point (targetX, targetY)
        // theta: Barnes-Hut accuracy parameter (e.g., 0.5)
        public void ComputeRepulsiveForce(double targetX, double targetY, double theta, ref double fx, ref double fy)
        {
            double dx = CenterX - targetX;
            double dy = CenterY - targetY;
            double distSq = dx * dx + dy * dy + 1e-8; // avoid div by zero
            double width = Math.Max(MaxX - MinX, MaxY - MinY);
            // Barnes-Hut criterion: if width / sqrt(distSq) < theta, treat as single body
            if (IsLeaf || (width / Math.Sqrt(distSq) < theta))
            {
                // Repulsive force: Count / distSq * dx, dy
                fx += Count * dx / distSq;
                fy += Count * dy / distSq;
            }
            else if (Children != null)
            {
                foreach (var child in Children)
                {
                    if (child != null)
                        child.ComputeRepulsiveForce(targetX, targetY, theta, ref fx, ref fy);
                }
            }
        }

        // Build a quadtree for points Y (n x 2)
        public Quadtree(double[,] Y, List<int> indices, double minX, double minY, double maxX, double maxY, int maxLeaf = 1)
        {
            MinX = minX; MinY = minY; MaxX = maxX; MaxY = maxY;
            Points = new List<int>(indices);
            Count = indices.Count;
            if (Count <= maxLeaf)
            {
                // Leaf node: compute center of mass
                double cx = 0, cy = 0;
                foreach (var idx in indices)
                {
                    cx += Y[idx, 0];
                    cy += Y[idx, 1];
                }
                CenterX = cx / Count;
                CenterY = cy / Count;
                Children = null;
            }
            else
            {
                // Split into 4 quadrants
                Children = new Quadtree?[4];
                double midX = 0.5 * (minX + maxX);
                double midY = 0.5 * (minY + maxY);
                var q0 = new List<int>(); // top-left
                var q1 = new List<int>(); // top-right
                var q2 = new List<int>(); // bottom-left
                var q3 = new List<int>(); // bottom-right
                foreach (var idx in indices)
                {
                    double x = Y[idx, 0];
                    double y = Y[idx, 1];
                    if (x < midX && y < midY) q0.Add(idx);
                    else if (x >= midX && y < midY) q1.Add(idx);
                    else if (x < midX && y >= midY) q2.Add(idx);
                    else q3.Add(idx);
                }
                Children[0] = q0.Count > 0 ? new Quadtree(Y, q0, minX, minY, midX, midY, maxLeaf) : null;
                Children[1] = q1.Count > 0 ? new Quadtree(Y, q1, midX, minY, maxX, midY, maxLeaf) : null;
                Children[2] = q2.Count > 0 ? new Quadtree(Y, q2, minX, midY, midX, maxY, maxLeaf) : null;
                Children[3] = q3.Count > 0 ? new Quadtree(Y, q3, midX, midY, maxX, maxY, maxLeaf) : null;
                // Center of mass for internal node
                double cx = 0, cy = 0;
                int total = 0;
                foreach (var child in Children)
                {
                    if (child != null)
                    {
                        cx += child.CenterX * child.Count;
                        cy += child.CenterY * child.Count;
                        total += child.Count;
                    }
                }
                CenterX = cx / total;
                CenterY = cy / total;
                Count = total;
            }
        }
    }
}
