import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline
import os
from fpdf import FPDF
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# Check if Plot.csv exists before proceeding
if not os.path.exists("Plot.csv"):
    print("Error: Plot.csv not found. Ensure it is generated first.")
    exit(1)

# Load data
df = pd.read_csv("Plot.csv")

# Extract columns
percent_gc = df["% GC"].values
reference_values = df["Reference"].values
sample_columns = df.columns[3:]  # All columns after 'Reference'

# Create output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Define 4 knots at evenly spaced percentiles (to match Prism)
knots = np.percentile(percent_gc, [20, 40, 60, 80])

# Store computed areas
area_data = []
plot_files = []

# Generate plots
for sample in sample_columns:
    sample_values = df[sample].values
    ref_spline = LSQUnivariateSpline(percent_gc, reference_values, knots)
    sample_spline = LSQUnivariateSpline(percent_gc, sample_values, knots)

    x_smooth = np.linspace(min(percent_gc), max(percent_gc), 310384)
    ref_smooth = ref_spline(x_smooth)
    sample_smooth = sample_spline(x_smooth)
    
    mask = sample_smooth < ref_smooth
    x_below = x_smooth[mask]
    y_diff_below = ref_smooth[mask] - sample_smooth[mask]
    area_below = np.trapezoid(y_diff_below, x_below) if len(x_below) > 0 else 0
    
    area_data.append({"Sample": sample, "Area_Below_Reference": area_below})
    
    plt.figure(figsize=(8, 6))
    plt.scatter(percent_gc, sample_values, color="blue", alpha=0.5, s=10)
    plt.scatter(percent_gc, reference_values, color="lightgrey", alpha=0.5, s=10)
    plt.plot(x_smooth, ref_smooth, label="Reference", color="black", linestyle="--", linewidth=2)
    plt.plot(x_smooth, sample_smooth, label=sample, color="blue", linewidth=2)
    plt.xlim(25, 90)
    plt.ylim(-5, 15)
    plt.xlabel("% GC")
    plt.ylabel("Expression")
    plt.title(f"Spline Fit: {sample} vs Reference")
    plt.legend()
    plot_filename = os.path.join(output_dir, f"{sample}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_filename)

# Save area data
area_df = pd.DataFrame(area_data)
area_df.to_csv("area_below_reference.csv", index=False)

# === Outlier Detection ===
area_values = area_df["Area_Below_Reference"].values.reshape(-1, 1)
rlm_model = RLM(area_values, np.ones_like(area_values), M=HuberT())
rlm_results = rlm_model.fit()
residuals = area_values.flatten() - rlm_results.fittedvalues
p_values = 2 * (1 - norm.cdf(np.abs(residuals) / rlm_results.scale))
_, adjusted_pvals, _, _ = multipletests(p_values, alpha=0.01, method="fdr_bh")
area_df["Adjusted P-Value"] = adjusted_pvals
area_df["Outlier"] = adjusted_pvals < 0.01
area_df.to_csv("area_below_reference.csv", index=False)

# === Modify Plots to Add Delta Area Text ===
for _, row in area_df.iterrows():
    sample = row["Sample"]
    area_below = row["Area_Below_Reference"]
    plot_filename = os.path.join(output_dir, f"{sample}.png")
    plt.figure(figsize=(8, 6))
    img = plt.imread(plot_filename)
    plt.imshow(img)
    plt.axis("off")
    x_center = img.shape[1] / 2
    y_position = int(img.shape[0] * 0.85)
    plt.text(
        x_center, y_position,
        f"Delta area = {area_below:.2f}",
        fontsize=10,
        color="black",
        ha="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

# === Regenerate PDF with Updated Plots ===
pdf = FPDF()
for img_file in plot_files:
    pdf.add_page()
    pdf.image(img_file, x=10, y=10, w=180)
pdf.output("spline_plots.pdf")
