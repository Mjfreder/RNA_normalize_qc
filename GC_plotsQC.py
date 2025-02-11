import os
import subprocess
import sys

# Get the PDM virtual environment's Python interpreter
pdm_python = os.path.join(os.path.dirname(sys.executable), "python")

print("Generating Plot.csv...")
subprocess.run([pdm_python, "generate_plot_csv.py"], check=True)

if os.path.exists("Plot.csv"):
    print("Running analysis on Plot.csv...")
    subprocess.run([pdm_python, "analyze_plot_csv.py"], check=True)
else:
    print("Error: Plot.csv not generated.")
