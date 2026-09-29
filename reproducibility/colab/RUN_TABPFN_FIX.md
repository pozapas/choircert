# TabPFN rerun with the context fix

The earlier TabPFN run had no fatal (K) records in its 10,000-record context, so it predicted
P(K) = 0 for every record. `run_tabpfn_fixed.py` fixes the sampler and runs TabPFN only.

## Steps

1. In Google Drive, open `MyDrive/CHOIR/` and make sure it contains:
   - `analysis.parquet` (the current snapshot, 2026-07-11)
   - `colab_common.py`
   - `tabpfn_token.txt` (your PriorLabs key)
   - `run_tabpfn_fixed.py` (upload it from this folder)
2. Open a Colab notebook with a GPU runtime (A100 if available) and run these cells:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/CHOIR
   !pip -q install tabpfn polars pyarrow scikit-learn
   %run run_tabpfn_fixed.py
   ```

3. Check the printed output:
   - `fingerprint 416bdd0ac68fbfec`. If it differs, stop: the snapshot on Drive is not the current one.
   - `context composition by KABCO class:` should show all five classes, with at least 200 for classes 2 to 5.
   - `mean P(K) on test:` must be greater than 0.
4. Download `colab_outputs/s1_tabpfn_fixed.npz` and `colab_outputs/s1_tabpfn_fixed_meta.json`
   and send them back. The local analysis will rename the npz to `s1_tabpfn.npz` and rerun the
   TabPFN rows of the tables.

Expected time: the fit takes seconds; prediction over about 1.6 million calibration and test
records takes roughly 15 to 70 minutes depending on the GPU.
