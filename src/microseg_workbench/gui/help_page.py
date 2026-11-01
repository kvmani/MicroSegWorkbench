"""In-app user guide."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

GUIDE = """
<h2>MicroSegWorkbench - user guide</h2>
<p>The workbench takes you from annotated micrographs to a trained, evaluated segmentation model.
Work through the steps on the left in order. Every step can also be run from the command line (<code>msw</code>).</p>

<h3>Finding your way</h3>
<ul>
<li>The steps on the left show where you are: <b>&#10003;</b> done, <b>&#9888;</b> needs your attention, <b>&#8226;</b>
to do, <b>&#9702;</b> optional, grey = not possible yet (hover for the reason).</li>
<li>The bar at the bottom says <i>what to do here</i> and has <b>Next</b> and <b>Back</b> buttons.</li>
<li><b>Beginner</b> mode (top right) shows presets and visual controls only. <b>Advanced</b> adds every option with
exact values. The choice is remembered for this workspace.</li>
<li>Before training, the <b>Ready to train?</b> list in step 3 checks the dataset, preprocessing, augmentation check
and device. A red &#10007; must be fixed first; the others are advice.</li>
</ul>

<h3>Try it in two minutes</h3>
<ol>
<li><b>1 Data</b>: press <i>Create demo data</i>. Synthetic micrographs (not real data) are written and checked.</li>
<li>Press <i>Create dataset version</i>.</li>
<li><b>2 Preprocess</b>: pick <i>Standard</i> and look at the training patches at the bottom.</li>
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

<h3>Importing from OnlineAnnotator</h3>
<p>In step 1 press <i>Import OnlineAnnotator export...</i> and choose the ZIP downloaded from OnlineAnnotator (any
layout and mask style). Every file is checked against the fingerprints in the export, class numbers stay exactly
as in OnlineAnnotator, and who annotated and approved each mask is kept with the dataset. Masks that were submitted
but not approved are flagged, and results trained on them say so. If splits were assigned in OnlineAnnotator, choose
<i>Split: As assigned in OnlineAnnotator</i>; this is refused if images of one specimen ended up in different
splits.</p>

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
<i>Per-image standardization</i> removes overall brightness and contrast differences between acquisitions.
<i>Exclude border</i> removes a strip at the image edges (for example an SEM information bar) before anything
else: it is never trained on or scored, and predicted masks then cover only the analysed region (its position is
recorded in <code>predictions.json</code>).</p>

<h3>The augmentation pipeline (step 2)</h3>
<p>Augmentation shows the model many slightly different versions of each training image, so it learns the features
rather than one specimen's look. Step 2 draws the pipeline as blocks in the order they run, top row first:</p>
<ul>
<li><b>Blue blocks</b> (geometry) move image and mask together - the coloured mask always stays on the features.</li>
<li><b>Orange blocks</b> (intensity) change only the image; they run as one group with a shared <i>Apply intensity
changes to</i> setting.</li>
<li>Each block shows its weakest and strongest effect on your own image. <i>How often</i> is the chance it is used on
a patch (the dots show it out of 10); <i>How strong</i> is the largest change.</li>
<li><b>Start from</b> None, Gentle, Standard or Strong, then adjust. <b>My settings</b> saves and loads your own
combinations in this workspace.</li>
<li><b>Training patches</b> shows 12 random patches exactly as the model will see them; click one and open
<b>Step by step</b> to follow it through each block. <b>Selected block</b> shows one block's effect larger.</li>
<li><b>Check settings</b> runs the real pipeline on 200 patches and warns if brightness/contrast clip pixels to pure
black or white, if too many pixels come from outside the image, if blur is wider than your thinnest features, or if
blocks fire more or less often than set.</li>
</ul>
<p>The order of blocks is fixed on purpose: geometry first, then intensity, so masks stay aligned.</p>

<h3>Extra augmentations</h3>
<p>Your developers can add more image-only operations as plugins (see <code>examples/plugins/example_ops.py</code>);
they then appear as extra orange blocks at the end of the pipeline. Plugins never change masks, and every run records
which plugin files it used.</p>

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

<h3>Large datasets</h3>
<p>By default all training images are kept in memory. If they do not fit, training stops at once and suggests a
value for <i>Images in memory</i>; with a number set, images are processed in blocks of that size and validation
and test images are read one at a time. The setting is saved with the run, so the same run can be reproduced
on any computer.</p>

<h3>Early stopping</h3>
<p><i>Early stopping after N epochs</i> ends training when the validation score has not improved for N epochs.
The best checkpoint is evaluated on the test split as usual and the run notes that it stopped early.</p>

<h3>Stopping and resuming</h3>
<p>A cancelled run, a failed run or a run whose computer crashed keeps the checkpoint of its last completed epoch.
Select it in <b>4 Results</b> and press <i>Resume training</i>; it continues exactly where it stopped.</p>

<h3>Sharing a result as a report</h3>
<p>In <b>4 Results</b>, <i>Export report</i> writes <code>evaluation/report.html</code>: one self-contained page with
the review status, test scores per class, training curves, test overlays, all settings and provenance. It opens
in any browser without network access and can be emailed or printed.</p>

<h3>Comparing runs</h3>
<p><b>5 Compare</b> shows ticked runs side by side: scores, settings that differ, validation curves and the same
test image predicted by each run. Only runs on the same dataset version share a test set; otherwise a warning
appears and their test scores must not be ranked.</p>

<h3>How much does luck matter? (repeat with seeds)</h3>
<p>Small datasets and rare classes can make results depend on the random seed. Set <i>Repeat with seeds</i> in step 3
to train several runs that differ only in their seed; tick them in <b>5 Compare</b> to see the mean and spread of
the test scores. Only differences larger than that spread should be trusted.</p>

<h3>Segmenting new images</h3>
<p>In <b>4 Results</b>, select a run and press <i>Segment new images...</i>. Choose a folder of micrographs and a new
output folder; each image gets a mask (<code>_pred_mask.png</code>, original class values), an overlay and an entry in
<code>predictions.json</code> with the model's identity. Predictions are unreviewed.</p>

<h3>Results and approval</h3>
<p>The best checkpoint (highest validation feature IoU) is evaluated on the test split. Results stay
<b>unreviewed</b> until a colleague reviews the published run in <b>6 Share</b>.</p>

<h3>Feature statistics (hydrides)</h3>
<p>In <b>4 Results</b>, the <i>Feature statistics</i> tab compares ground truth and prediction on the test images for
one class: area fraction, number of features, and how different the size and orientation distributions are
(Kolmogorov-Smirnov distance, 0 = identical). The calculations are HydrideSegmentation's own; the tab says so and
asks for that package if it is not installed. Enter micrometres per pixel to get physical sizes.</p>

<h3>Working with colleagues (6 Share)</h3>
<ul>
<li>Enter the shared office folder once (an administrator initialises a new one with <i>Initialise new shared
folder</i>). Your name is recorded with everything you publish or review.</li>
<li><b>Publish</b> a dataset version before publishing runs trained on it, so anyone can reproduce them. Only
finished runs can be published; the best checkpoint, settings, logs, sample predictions and test results go
with it.</li>
<li>Published items never change. Files are stored once even if published twice, every copy is checked against
its fingerprint (SHA-256), and an interrupted download continues where it stopped.</li>
<li><b>Download</b> a colleague's dataset to train on it locally, or a run to compare it or segment images with it.</li>
<li><b>Review</b>: choose approved, needs changes or rejected, and say what you checked. You cannot approve your
own model. The latest review decides the status; <i>Approved models only</i> lists deployment candidates.</li>
<li>If an optional metadata service is configured (address and personal token from the administrator), it is
updated automatically and confirms who submitted each record.</li>
</ul>

<h3>Where files are stored</h3>
<p>Everything lives in the workspace folder shown at the top of the window: <code>datasets/</code> holds dataset
versions, <code>runs/</code> holds one folder per training run (settings, live log, checkpoints, sample
predictions and evaluation) and <code>cache/</code> holds downloaded files from the shared folder. Keep the workspace
on a local disk; only the shared folder lives on the network drive.</p>
"""


class HelpPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.browser = QTextBrowser()
        self.browser.setHtml(GUIDE)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)
