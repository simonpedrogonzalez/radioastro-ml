### Imaging setup comparison

| Setting        | Regular              | Reproduction         | Change |
|----------------|----------------------|----------------------|--------|
| Cell (arcsec)  | 2.793                | 1.855                | smaller pixels |
| FoV (arcsec)   | 714.925              | 474.788              | smaller FoV |
| Beam (major)   | 26.558               | 26.318               | same |
| Beam (minor)   | 11.270               | 11.240               | same |

---

### Imaging and display procedure (high-level)

- A first-pass beam is estimated from array configuration and observing frequency.  
- The cell size is chosen so that the synthesized beam minor axis is sampled by a fixed number of pixels (here, 4 pixels per beam).  
- The field of view is set to span a fixed number of beam widths (here, 64 times the beam minor axis).  
- The image size is computed from this target FoV and cell size.  
- A first-pass dirty image is made, and the actual restoring beam is measured from it.  
- The final imaging grid (cell and FoV) is recomputed using this measured beam.  

For visualization:
- Images are converted from Jy/beam to mJy/beam.  
- Display limits are set using percentiles of the pixel values.  
- Typically, the range is symmetric around zero and clipped at the 99.5th percentile of the absolute pixel values.  

---

### QA metrics comparison

| Metric | Regular | Reproduction | Ratio (repro/reg) |
|--------|--------|--------------|-------------------|
| σ      | 0.0005 | 0.0005       | 0.9814            |
| max    | 4.6157 | 4.2369       | 0.9179            |
| p99    | 3.0735 | 3.2801       | 1.0672            |
| p995   | 3.5723 | 3.7184       | 1.0409            |
| DR     | 2010.4010 | 2047.8820 | 1.0186            |

Mixed: slightly better noise, peak residual, and DR, slightly worse residual tails.