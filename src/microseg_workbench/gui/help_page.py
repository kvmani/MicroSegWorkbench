"""In-app user guide."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

GUIDE = """
<h2>MicroSegWorkbench - user guide</h2>
<p>The workbench takes you from annotated micrographs to a trained, evaluated segmentation model.
Work through the steps on the left in order. Every step can also be run from the command line (<code>msw</code>).</p>

<h3>Try it in two minutes</h3>
<ol>
<li><b>1 Data</b>: press <i>Create demo data</i>. Synthetic micrographs (not real data) are written and checked.</li>
<li>Press <i>Create dataset version</i>.</li>
<li><b>2 Preprocess</b>: press <i>Preview</i> to see random flips, rotations and brightness changes.</li>
<li><b>3 Training</b>: keep the <i>Quick CPU</i> preset and press <i>Start training</i>.</li>
<li><b>4 Results</b>: inspect test-set metrics and overlays.</li>
</ol>

<h3>Preparing your own folder</h3>
<ul>
<li>One micrograph per file: <code>S01_0001.png</code> (PNG, TIFF, JPEG or BMP; 8- or 16-bit; grayscale or RGB).</li>
<li>Its mask next to it: <code>S01_0001_mask.png</code> - same width and height, lossless PNG.</li>
<li>A <code>classes.json</code> naming every mask value, for example:
<pre>{"schema": "microseg-workbench.classes/1",
 "mask_encoding": "indexed",
 "classes": [{"value": 0, "name": "matrix",  "color": "#000000"},
             {"value": 1, "name": "hydride", "color": "#e6194b"}]}</pre>
Use <code>"mask_encoding": "binary_0_255"</code> for black/white masks where 255 means the feature
(classes must then be values 0 and 1). Folders exported by OnlineAnnotator in the <i>pairs</i> layout with indexed
or binary masks follow the same naming; add a classes.json describing the export.</li>
</ul>
<p>Masks with values not listed in classes.json, colour masks, mismatched sizes, missing partners and duplicate images
are reported as errors. Nothing is silently converted, thresholded or dropped.</p>

<h3>Groups and leakage</h3>
<p>Images from the same specimen or micrograph look alike. If some are in training and others in the test set,
test scores are too optimistic. The <i>group pattern</i> extracts a group name from each file name (default: the
text before the first underscore) and every group goes entirely into one split. Check the group column and the
split preview before creating a dataset version.</p>

<h3>Dataset versions</h3>
<p>A dataset version is a frozen copy of the images, masks, class map and split, identified by a content hash.
It never changes; editing data means creating a new version. The test split is therefore fixed per version.</p>

<h3>Reading training progress</h3>
<ul>
<li><b>Loss</b> should fall for both training and validation. If validation loss rises while training loss falls,
the model is overfitting.</li>
<li><b>Validation mean IoU</b> (0 to 1) measures overlap between predicted and annotated features.</li>
<li>The <b>prediction evolution</b> panel shows the model's output for the same validation images after each epoch.</li>
<li>CPU, memory and GPU figures are real measurements of the training process.</li>
</ul>

<h3>Preprocessing</h3>
<p><i>Resize factor</i> shrinks or enlarges images before the model sees them (0.5 halves width and height). Masks are
resized with nearest neighbour, so thin features can vanish when shrinking: check the preview. Evaluation and
predictions are always mapped back to the original resolution and compared with the original masks.
<i>Per-image standardization</i> removes overall brightness and contrast differences between acquisitions.</p>

<h3>GPU training</h3>
<ul>
<li>The <b>Device</b> list and the line below it show whether an NVIDIA GPU was found and why not, if not.</li>
<li><b>Automatic batch size</b> tries 64, 32, 16 ... images per step until one fits in GPU memory.</li>
<li><b>Mixed precision</b> (automatic on GPUs) roughly halves memory use and speeds up training.</li>
<li>If the GPU runs out of memory the run stops with advice (smaller batch size or patches); nothing is lost
before the last completed epoch.</li>
<li>The <i>Computer resources</i> tab shows CPU, RAM, GPU utilisation and GPU memory. Values marked SIMULATED come
from a test mode and are not real.</li>
</ul>

<h3>Stopping and resuming</h3>
<p>A cancelled run, a failed run or a run whose computer crashed keeps the checkpoint of its last completed epoch.
Select it in <b>4 Results</b> and press <i>Resume training</i>; it continues exactly where it stopped.</p>

<h3>Comparing runs</h3>
<p><b>5 Compare</b> shows ticked runs side by side: scores, settings that differ, validation curves and the same
test image predicted by each run. Only runs on the same dataset version share a test set; otherwise a warning
appears and their test scores must not be ranked.</p>

<h3>Segmenting new images</h3>
<p>In <b>4 Results</b>, select a run and press <i>Segment new images...</i>. Choose a folder of micrographs and a new
output folder; each image gets a mask (<code>_pred_mask.png</code>, original class values), an overlay and an entry in
<code>predictions.json</code> with the model's identity. Predictions are unreviewed.</p>

<h3>Results and approval</h3>
<p>The best checkpoint (highest validation feature IoU) is evaluated on the test split. All results are marked
<b>unreviewed</b>; a human approval workflow will be added before models can be published for use.</p>

<h3>Where files are stored</h3>
<p>Everything lives in the workspace folder shown at the top of the window: <code>datasets/</code> holds dataset
versions and <code>runs/</code> holds one folder per training run (settings, live log, checkpoints, sample
predictions and evaluation).</p>
"""


class HelpPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.browser = QTextBrowser()
        self.browser.setHtml(GUIDE)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)
