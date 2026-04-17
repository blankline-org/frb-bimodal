"""
Reproduce the bimodal drift rate analysis end-to-end.

Runs the four analysis scripts in order and regenerates all figures
under results/third_species/.

Usage:
    python reproduce.py
"""
import subprocess
import sys

SCRIPTS = [
    "discover_cluster.py",
    "characterize_cluster.py",
    "robustness.py",
    "verify_bimodality.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll scripts completed. Figures written to results/third_species/.")


if __name__ == "__main__":
    main()
