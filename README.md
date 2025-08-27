# Polynomial Curve Fitting and K-Means Clustering from Scratch

This repository contains the solution to Homework 1 of the *Introduction to Machine Learning* course at National Chung Cheng University, instructed by Prof. Peggy Lu.

## Directory Structure

```
.
├── HW1\_413430901\_matthis.pdf       # Main report with detailed solutions
├── ML\_hw1.pdf                      # Exercise description or additional notes
├── Hw1-1/                          # Problem 1: Polynomial Curve Fitting
│   ├── a.py                        # Part (a): Polynomial fit (m = 2)
│   ├── b.py                        # Part (b): Best-fit polynomial order
│   ├── c.py                        # Part (c): Comparison (m = 3 vs m = 8)
│   ├── main.py                     # Runs all subparts
│   ├── x.txt                       # Input dataset (features)
│   ├── y1.txt                      # Input dataset (labels)
├── Hw1-2/                          # Problem 2: Clustering
│   ├── a.py                        # Part (a): K-Means with initialization A
│   ├── b.py                        # Part (b): K-Means with initialization B
│   ├── c.py                        # Part (c): Non-uniform binary split
│   ├── main.py                     # Runs all clustering parts
│   ├── Dataset\_2.txt               # Input dataset
├── Hw1-2\_TA\_explanation.pptx       # Optional TA slides (reference)

```

## Problem 1: Polynomial Curve Fitting

Implemented from scratch:
- (a) Polynomial fitting with degree m = 2  
- (b) Selection of the best-fit polynomial order (visual and quantitative analysis)  
- (c) Comparison of regression performance for m = 3 and m = 8  

**Datasets:** `x.txt`, `y1.txt`  
**Outputs:** Plots and error analysis (included in the report)  
**Restriction:** no use of built-in fitting functions such as `numpy.polyfit`.  

## Problem 2: Clustering

Implemented from scratch:
- (a) K-Means (K=5) with centroid initialization A  
- (b) K-Means (K=5) with centroid initialization B  
- (c) Non-uniform binary split clustering  

**Dataset:** `Dataset_2.txt`  
**Outputs:** Cluster visualizations (included in the report)  

**Cluster color convention:**
- Cluster 1: Blue  
- Cluster 2: Black  
- Cluster 3: Red  
- Cluster 4: Green  
- Cluster 5: Magenta  

## Notes

- All scripts are written in pure Python, without high-level ML libraries.  
- To reproduce results:  
  - Run `main.py` inside `Hw1-1/` for Problem 1  
  - Run `main.py` inside `Hw1-2/` for Problem 2  