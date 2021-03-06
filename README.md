# PSGPy
Processing, analysis and visualization of Polysomnography data

## Requirements
Core functionality is implemented using:
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`


For reading EDF files and scored hypnograms,
- `MNE` must be correctly installed

## Demo
**Step 0. Clone repository**
```bash
git clone git@github.com:pranaysy/PSGPy.git
cd PSGPy
```
After entering the `PSGPy` directory, launch an interpreter inside an
environment with the required packages installed.

### Read and visualize a Hypnogram

**Step 1. Import call**
```python
import PSGPy
from pathlib import Path
```


**Step 2. Read hypnogram**
```python
file = Path("/path/to/hypnogram.edf")
hypno = PSGPy.load_hypnogram(file, wake_threshold=2)
```
The parameter `wake_threshold` indicates cutoff in minutes for short and long awakenings.


**Step 3. Plot hypnogram**
```python
# Resample hypnogram to epoch units of time
hypno_resampled = PSGPy.resample_hypnogram(hypno)

# Plot hypnogram
fig, ax = PSGPy.plot_hypnogram(hypno_resampled)
```

![Example Hypnogram](Hypnogram.jpg)

**Step 4. Save hypnogram**
```python
PSGPy.save_hypnogram_plot(fig, label="Example", folder=Path("/output/plots/"))
```
Output formats can be selected as additional optional arguments, by default `.tiff` with
LZMA compression is preferred, alternatives are `.jpg` and `.svg`. See docstrings for more.

### Detect sleep cycles and visualize

**Detect cycles using deterministic criteria**
```python
cycles = PSGPy.detect_cycles(hypno, min_length=10, min_separation=10)
```
Two parameters indirectly control onset and offset of each cycle.
- `min_length` indicates the minimum duration of consecutive NREM runs for cycle onset
- `min_separation` indicates the gap between two NREM runs to qualify as distinct cycles


Offset is decided by three fail-over fixed critieria:
1. Offset of last REM after an NREM run
2. Offset of last N3 after an NREM run, if followed by a short awakening
3. Onset of first long awakening after an NREM run


**Visualize cycles**
```python
# Update hypnogram with cycle information
hypno = PSGPy.update_hypnogram_cycles(hypno, cycles)

# Resample hypnogram to epoch units of time
hypno_resampled = PSGPy.resample_hypnogram(hypno)

# Plot hypnogram with cycles marked
fig, ax = PSGPy.plot_hypnogram(hypno_resampled, cycles)
```
![Example Hypnogram with Cycles](Hypnogram_with_Cycles.jpg)
