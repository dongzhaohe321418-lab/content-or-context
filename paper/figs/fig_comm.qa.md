# Figure 1 (fig_comm) QA record
- Drawn at the manuscript text width (5.5 in); included at \linewidth, scale 1.00. Smallest glyph in the compiled
  paper: 5.5 pt (PDF text audit PASS, floor 5 pt).
- Static validator: 18 pass, 3 warn (bootstrap RNG flagged as possible simulated data - it resamples real test
  articles; log axis - values are clipped to the axis floor; raster preview), 0 fail.
- Panel alignment (render-time, 1.5 pt): PASS, no exemptions. Panel a uses its own grid (wider left margin for
  condition labels); panels b and c share a grid. Panel labels are placed at a fixed point offset from each axes.
- Collision audit: 0 FAIL; 3 WARN (text-fill-edge): legend labels inside the bounding boxes of confidence-band
  polygons (panel b "Text"; panel c "Text", "LLM text (comm)"). Inspected at final size: band pixels lie well away
  from the legend text. Accepted.
- Statistics: panel a values and content intervals from pb_decompose.py on 493 windows (inside one article, with a
  same-article donor); five donor draws of each kind; bootstrap over articles, fitting seeds and donor draws (4000).
  Panels b and c: all 512 windows, first other-article donor draw from pb_mismatch.py; pointwise 95% bands from 4000
  article-clustered draws, conditional on the saved fits and that donor draw. Source data incl. band endpoints:
  fig_comm.source.json.
