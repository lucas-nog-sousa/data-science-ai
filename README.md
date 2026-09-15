# data-science-ai
# Handling Heavily Skewed Operational Metrics: Non-Parametric & Bootstrap Analysis

This directory contains the statistical workflows, Python implementation, and visualization scripts for analyzing right-skewed operational logs (such as Zendesk support ticket handling times, first response times, and transfer counts).

It accompanies the technical article: **["Handling Heavily Skewed Operational Metrics: Why Non-Parametric Tests and Bootstrapping Beat Standard t-Tests"](https://medium.com/@lucasnogsousa)**.

---

## Overview

Operational support metrics rarely follow a Gaussian distribution. They are heavily right-skewed, characterized by a large volume of quick resolutions and a long tail of complex tickets. 

Standard parametric tests (e.g., Welch's $t$-test) rely on sample means and variances that are easily inflated by extreme outliers, leading to inaccurate $p$-values and misleading operational targets. This project provides a robust alternative framework using **rank-based non-parametric tests (Mann-Whitney U)** and **computational bootstrapping**.

---

## Key Features

* **Log-Normal Data Simulation:** Generates synthetic support duration logs replicating realistic active vs. churned customer behavior.
* **Distribution Diagnosis:** Calculates skewness metrics and renders high-density KDE plots with explicit mean vs. median divergence.
* **Parametric vs. Non-Parametric Benchmark:** Runs Welch's $t$-test alongside the Mann-Whitney U test and calculates Rank-Biserial Correlation ($r$) for effect size.
* **Median Bootstrapping Pipeline:** Implements non-parametric resampling ($10,000$ iterations) to generate $95\%$ confidence intervals for median differences.

---

## Repository Structure

```text
data-science-ai/
├── main.py                     # Primary statistical pipeline
├── plot_distribution.py        # Visual script for right-skewed KDE plots
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
