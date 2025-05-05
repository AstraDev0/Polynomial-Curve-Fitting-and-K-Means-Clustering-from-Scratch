```markdown
# ML-HW1: Polynomial Curve Fitting and K-Means Clustering from Scratch

This repository contains the solution to Homework 1 of the "Introduction to Machine Learning" course at National Chung Cheng University, instructed by Prof. Peggy Lu.

## 📁 Directory Structure

```

HW1\_StudentID\_Name/
│
├── HW1\_StudentID\_Name.pdf         # Report with problem statements and detailed solutions
│
├── Hw1-1/                         # Problem 1: Polynomial Curve Fitting
│   ├── polynomial\_fit.py          # Main script for polynomial fitting
│   ├── plot\_m2.png                # Plot for m=2 fit
│   ├── plot\_m\_best.png            # Plot for best-fit polynomial
│   ├── plot\_m3\_m8\_errors.png      # Comparison of errors for m=3 and m=8
│   └── ...
│
├── Hw1-2/                         # Problem 2: Clustering Algorithms
│   ├── kmeans\_case\_a.py           # K-means with specified initialization set A
│   ├── kmeans\_case\_b.py           # K-means with initialization set B
│   ├── binary\_split.py            # Non-Uniform Binary Split clustering
│   ├── plot\_case\_a.png            # Clustering result for (a)
│   ├── plot\_case\_b.png            # Clustering result for (b)
│   ├── plot\_binary\_split.png      # Clustering result for (c)
│   └── ...

```

## 🧪 Problem 1: Polynomial Curve Fitting

This task involves estimating the mapping function from input to output data using polynomial regression.

- ✅ Fit polynomial of order **m = 2**
- ✅ Find and plot the best-fitting polynomial
- ✅ Compare **m = 3** and **m = 8** using average sum of squared errors (SSE)
- ✅ Visualize dataset and regression curves
- 🔧 All fitting is done **from scratch** (no `numpy.polyfit`, etc.)

## 🧪 Problem 2: Clustering

This problem requires unsupervised learning algorithms implemented manually.

- ✅ **K-Means (K = 5)** with 2 different centroid initializations
- ✅ **Non-Uniform Binary Split algorithm** for clustering
- ✅ Track and plot distortion function after each E-step and M-step
- ✅ Visualize clustering results using fixed color mapping:
  - Cluster 1: Blue
  - Cluster 2: Black
  - Cluster 3: Red
  - Cluster 4: Green
  - Cluster 5: Magenta
- 🔧 No use of `sklearn.cluster.KMeans`, `scipy`, or similar libraries

## 📌 Notes

- All code is written in **pure Python**
- Figures are generated and saved automatically when scripts are run
- The `.pdf` file includes written answers, parameters, and figure references

## 👩‍🏫 Instructor
**Prof. Peggy Lu**  
Email: peggylu@cs.ccu.edu.tw

## 📩 Teaching Assistants
- **David Lee** (TA for Problem 1) – david20010603@alum.ccu.edu.tw  
- **Oscar Wu** (TA for Problem 2) – oscarwu217@alum.ccu.edu.tw

## 📅 Deadline
**2025/04/10 – 23:59**

---

