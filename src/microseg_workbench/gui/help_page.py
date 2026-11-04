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
<li>The <b>workflow map</b> across the top shows the whole path - images and masks, dataset version, preprocessing,
augmentation, training, evaluation, comparison, sharing - coloured by status. Click a stage to go there.</li>
<li>Before training, the blue <b>This run</b> card in step 3 says how many epochs, steps and patches will run, how big
the model is and roughly how long it will take on this computer (timed with one real training step).</li>
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

<h3>Looking at your data (step 1)</h3>
<p>After <i>Check folder</i>, the <b>Pictures</b> tab shows every image with its mask; pairs with problems have a red
frame and a &#9888; - hover to read the problem, or click a problem in the list to jump to its picture. The bars show
how much of all pixels each class covers and in how many images it occurs. The split panel shows each specimen group
as a tile in the Training, Validation or Test column; a group is never split across columns. The <b>List</b> tab has the
same information as a table.</p>

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

<h3>Preprocessing (step 2, tab A)</h3>
<p>Drag the dashed edges on the image to leave out a strip such as an SEM data bar, or type the pixel counts below
it. The resize slider shows the same small area before and after resizing at true pixel size, and the typical width
of each feature class; a warning appears when features would become thinner than about 1.5 pixels and break up.</p>
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

<h3>Automatic checks</h3>
<ul>
<li><b>Dataset check</b> (after creating a dataset version and in <i>Ready to train?</i>): a class missing from the
training images (this blocks training - the model could not learn it), classes missing from validation or test, very
few images, a test set from a single specimen, masks that look inverted, and unapproved annotations.</li>
<li><b>Run check</b> (step 4, the report and <code>msw check-run</code>): overfitting, still improving at the last
epoch, a class the model never finds, test scores much lower than validation, and classes with too few test pixels
to trust. Each finding says what to try next.</li>
</ul>

<h3>Choosing a network</h3>
<p>Step 3 lets you pick the <b>Network</b>: <i>U-Net</i> (convolutional; fast and reliable on small datasets - the best
first choice) or <i>SegFormer-lite</i> (a transformer that sees wider context; it usually needs more images and
epochs). Train both on the same dataset version and compare them in step 5 or on single images. Developers can add more
networks as plugins; every run records which network (and plugin file) it used.</p>

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

<h3>Same image, several models</h3>
<p>In <b>5 Compare</b>, tick the runs (models) to compare, open <i>Same image, several models</i>, choose an image from a
dataset version (its mask is used as ground truth) or any image file, and press <i>Run ticked models on this image</i>.
You see each model's segmentation next to the image and the ground truth, a map of where the models disagree, and a
table with class fractions, time and - with ground truth - each model's scores on this image. Models with different
networks, presets or training data can be compared this way. Command line: <code>msw compare-image IMAGE RUN RUN ...
[--mask MASK --dataset DATASET] [--out side.png]</code>.</p>

<h3>Segmenting new images</h3>
<p>In <b>4 Results</b>, select a run and press <i>Segment new images...</i>. Choose a folder of micrographs and a new
output folder; each image gets a mask (<code>_pred_mask.png</code>, original class values), an overlay and an entry in
<code>predictions.json</code> with the model's identity. <code>predictions.csv</code> lists the area % of every class
in every image (opens in Excel) with any warnings. Predictions are unreviewed.</p>
<p>To use a colleague's model, select it in <b>6 Share</b> and press <i>Download and segment images...</i>: the model
and its dataset are fetched (verified), then your images are segmented. On the command line, <code>msw models</code>
lists every model here and on the shared folder, and <code>msw predict RUN ...</code> downloads a shared model
automatically.</p>
<p>The workbench remembers what the training images looked like (bit depth, brightness range). If a new image is,
for example, 16-bit where training used 8-bit, or clearly brighter or darker than every training image, it is still
segmented but you get a warning (also stored in <code>predictions.json</code>): check those results carefully or
retrain with such images.</p>
<p>The <i>Model card</i> tab in <b>4 Results</b> shows the model's identity card: network, classes, input
requirements, training data and annotation source, test scores, review status, software versions, checkpoint
fingerprint and observations. It is saved as <code>model_card.json</code> and published with the run.</p>

<h3>Results and approval</h3>
<p>The best checkpoint (highest validation feature IoU) is evaluated on the test split. Results stay
<b>unreviewed</b> until a colleague reviews the published run in <b>6 Share</b>.</p>

<h3>How accurate is the model? (Accuracy details)</h3>
<p>The <i>Per-class test metrics</i> tab gives pixel overlap per class: <b>IoU</b> (overlap / union, 1 = perfect),
Dice, precision (how much of what was predicted is right) and recall (how much of the truth was found). Judge a
model by the <b>feature classes</b> (for example hydride and pore): the matrix usually covers most pixels, so overall
pixel accuracy looks good even when features are missed.</p>
<p>The <i>Accuracy details</i> tab adds what matters for measurements:</p>
<ul>
<li><b>Boundary F1</b>: the share of boundary pixels lying within 2 px of the true boundary (1 = every edge in
place). Low boundary F1 with good IoU means areas are reliable but widths and shapes less so.</li>
<li><b>True and predicted area %</b> of each class on the test images and their difference in percentage points.
A clear difference means the model systematically over- or under-segments that phase, so area fractions
measured with it would be biased; the run check warns about this.</li>
<li><b>Weakest test images</b>: look at their overlays first; one bad image may be unusual or have a wrong mask.
All per-image scores are in <code>evaluation/per_image.csv</code>.</li>
<li><b>Where pixels go</b>: for each true class, the % of its pixels predicted as each class (confusions).</li>
</ul>
<p>All scores are computed at the original resolution on test images the model never saw, excluding ignored mask
pixels and any excluded border. Few test specimens make every score uncertain; repeat with seeds and add specimens
before trusting small differences.</p>

<h3>Checking an installation</h3>
<p>Press <i>Check this installation</i> at the bottom of this page (or run <code>msw selftest</code>): in about a
minute the workbench trains, evaluates and segments tiny synthetic images on this computer, checks the GPU, the disk,
reproducibility and - if configured - the shared folder (inside a temporary folder that is removed again). The
report is saved in the workspace folder <code>selftest-reports</code>; IT uses it to sign off a workstation.</p>

<h3>Feature statistics (hydrides)</h3>
<p>In <b>4 Results</b>, the <i>Feature statistics</i> tab compares ground truth and prediction on the test images for
one class: area fraction, number of features, and how different the size and orientation distributions are
(Kolmogorov-Smirnov distance, 0 = identical). The calculations are HydrideSegmentation's own; the tab says so and
asks for that package if it is not installed. Enter micrometres per pixel to get physical sizes.</p>

<h3>Observations and the model card</h3>
<p>In step 4, the <b>Observations</b> tab keeps your notes about a run (what you checked, what works, what does not).
Notes are never edited or deleted - add a new one to correct an earlier one. Every finished run also has a
<code>model_card.json</code>: network, classes, input and preprocessing, training data and its annotation provenance,
all settings, software and computer, results, run check, observations and the checkpoint fingerprint. Publishing
requires at least one observation; notes and the model card are published with the run, and predictions record
which model card produced them.</p>

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
    def __init__(self, ctx=None) -> None:
        from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton

        super().__init__()
        self.ctx = ctx
        self.browser = QTextBrowser()
        self.browser.setHtml(GUIDE)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser, 3)
        row = QHBoxLayout()
        self.selftest_btn = QPushButton("Check this installation")
        self.selftest_btn.setToolTip("About a minute: trains, evaluates and segments tiny synthetic images here and "
                                     "tests the shared folder (in a temporary folder). For IT and first use.")
        self.selftest_btn.clicked.connect(self.run_selftest)
        row.addWidget(self.selftest_btn)
        row.addWidget(QLabel("Runs the self-test used for roll-out; the report is saved in the workspace."), 1)
        layout.addLayout(row)
        self.selftest_view = QPlainTextEdit()
        self.selftest_view.setReadOnly(True)
        self.selftest_view.setVisible(False)
        layout.addWidget(self.selftest_view, 2)
        self.selftest_result = None
        self.selftest_busy = False

    def run_selftest(self) -> None:
        from ..collab.settings import CollabSettings
        from ..selftest import run_selftest
        from .widgets import run_background

        if self.ctx is None or self.selftest_busy:
            return
        ws = self.ctx.workspace
        share = CollabSettings.load(ws).share_root()
        self.selftest_busy = True
        self.selftest_btn.setEnabled(False)
        self.selftest_view.setVisible(True)
        self.selftest_view.setPlainText("Running the self-test (about a minute)...")

        def done(result) -> None:
            self.selftest_busy = False
            self.selftest_btn.setEnabled(True)
            self.selftest_result = result
            text = result.text()
            if result.report_path:
                text += f"\nReport saved: {result.report_path}"
            self.selftest_view.setPlainText(text)

        def failed(message: str, details: str) -> None:
            self.selftest_busy = False
            self.selftest_btn.setEnabled(True)
            self.selftest_view.setPlainText(f"The self-test could not run: {message}\n{details}")

        run_background(lambda: run_selftest(ws, share=share), done, failed)
