# Visualization

This folder is intentionally separate from the training code.

Use the script below to generate a simple overview of the processed dataset:

```powershell
Set-Location 'c:\Nerdmap\Machine Learning\CubeML'
& 'c:/Nerdmap/Machine Learning/.venv/Scripts/python.exe' '.\visualization\dataset_overview.py'
```

The script reads `cfop-dataset-processed/dataset.pkl` and writes a single PNG file to `visualization/output/dataset_overview.png`.

The figure contains four plain summaries:

- next-move frequency
- histogram of misplaced stickers per state
- 6 most informative and 6 least informative tile positions by mutual information
- move share across simple scramble-complexity bins