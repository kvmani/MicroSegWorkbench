// MicroSegWorkbench beginner's guide - house slide style (black ribbons, Arial, text left, graphic right).
// Build: see tools/guide/README.md (screenshots -> crops -> this script). Output goes to the build folder.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const DIR = __dirname;
const BUILD = process.env.MSW_GUIDE_BUILD || path.join(DIR, "build");
const CROPS = path.join(BUILD, "crops");
const SIZES = JSON.parse(fs.readFileSync(path.join(CROPS, "manifest.json"), "utf8"));
const NUM = JSON.parse(fs.readFileSync(path.join(DIR, "numbers.json"), "utf8"));
const VERSION = process.env.MSW_GUIDE_VERSION || "1.0 release candidate";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.title = "MicroSegWorkbench - beginner's guide";
pres.author = "MicroSegWorkbench team";
// black and white only (greys inside graphics)
const C = { text1: "000000", background1: "FFFFFF", background2: "F2F2F2", accent2: "595959" };

const W = 13.333, TOP = 0.85, BOT_Y = 6.85, BOT_H = 0.65;
pres.defineSlideMaster({
  title: "RIBBONS",
  background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: TOP, fill: { color: "000000" }, line: { color: "000000", width: 0 } } },
    { rect: { x: 0, y: BOT_Y, w: W, h: BOT_H, fill: { color: "000000" }, line: { color: "000000", width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0, w: W - 1.2, h: TOP, align: "center",
      valign: "middle", fontFace: "Arial", fontSize: 28, bold: true, color: C.background1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "bottomline", type: "body", x: 0.9, y: BOT_Y, w: W - 1.8, h: BOT_H,
      align: "center", valign: "middle", fontFace: "Arial", fontSize: 17, color: C.background1, margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 0.75, y: BOT_Y + 0.17, w: 0.5, h: 0.3, color: "BFBFBF", fontFace: "Arial", fontSize: 10, align: "right" },
});

// ---------- helpers ----------------------------------------------------------------------------------------
const LEFT_X = 0.55, LEFT_W = 4.55, BODY_Y = 1.2, BODY_B = 6.6;
const RIGHT_X = 5.45, RIGHT_W = W - 5.45 - 0.5;

function badge(slide, n, x, y, d = 0.36, onImage = false) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.text1 },
    line: { color: C.background1, width: onImage ? 1.5 : 0 }, objectName: `badge ${n}` });
  slide.addText(String(n), { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: d > 0.33 ? 14 : 12,
    bold: true, color: C.background1, margin: 0, isTextBox: true });
}

// statements: [{n?, head, text?}]
function statements(slide, items, opts = {}) {
  const top = opts.y ?? BODY_Y + 0.1;
  const bottom = opts.bottom ?? BODY_B;
  const gap = opts.gap ?? 0.18;
  const heights = items.map((it) => {
    const narrow = it.n !== undefined;
    const headLines = Math.ceil(it.head.length / (narrow ? 34 : 38));
    const textLines = it.text ? Math.ceil(it.text.length / (narrow ? 42 : 47)) : 0;
    return 0.3 * headLines + 0.255 * textLines + 0.06;
  });
  let y = top;
  items.forEach((it, i) => {
    const hasN = it.n !== undefined;
    const tx = hasN ? LEFT_X + 0.5 : LEFT_X, tw = LEFT_W - (hasN ? 0.5 : 0);
    if (hasN) badge(slide, it.n, LEFT_X, y + 0.0);
    const runs = [{ text: it.head, options: { bold: true, fontSize: 16, color: C.text1, breakLine: !!it.text } }];
    if (it.text) runs.push({ text: it.text, options: { fontSize: 14, color: C.text1 } });
    slide.addText(runs, { x: tx, y, w: tw, h: heights[i], valign: "top", margin: 0, paraSpaceAfter: 2,
      fontFace: "Arial", isTextBox: true });
    y += heights[i] + gap;
  });
  if (y > bottom + 0.01) console.warn(`WARN statements overflow on slide ${opts.name}: ${y.toFixed(2)}`);
  return y;
}

function attention(slide, text, h = null) {
  const lines = Math.ceil(text.length / 40);
  const hh = h ?? 0.55 + lines * 0.255;
  const y = BODY_B - hh;
  slide.addShape(pres.shapes.RECTANGLE, { x: LEFT_X, y, w: LEFT_W, h: hh, fill: { color: C.background2 },
    line: { color: C.text1, width: 1.25 }, objectName: "attention box" });
  slide.addShape(pres.shapes.OVAL, { x: LEFT_X + 0.14, y: y + 0.13, w: 0.3, h: 0.3, fill: { color: C.text1 },
    line: { color: C.text1, width: 0 } });
  slide.addText("!", { x: LEFT_X + 0.14, y: y + 0.13, w: 0.3, h: 0.3, align: "center", valign: "middle",
    bold: true, fontSize: 14, color: C.background1, margin: 0, isTextBox: true });
  slide.addText([{ text: "ATTENTION", options: { bold: true, fontSize: 12, color: C.text1, breakLine: true,
    charSpacing: 1 } }, { text, options: { fontSize: 14, color: C.text1 } }],
  { x: LEFT_X + 0.58, y: y + 0.1, w: LEFT_W - 0.7, h: hh - 0.18, valign: "top", margin: 0, isTextBox: true });
  return y;
}

// Place an image (cropped screenshot) inside a box, aspect kept; returns placed rect
function picture(slide, name, box = {}) {
  const bx = box.x ?? RIGHT_X, by = box.y ?? BODY_Y, bw = box.w ?? RIGHT_W, bh = box.h ?? (BODY_B - BODY_Y);
  const [pw, ph] = SIZES[name];
  const s = Math.min(bw / pw, bh / ph);
  const w = pw * s, h = ph * s;
  const x = bx + (bw - w) / 2, y = by + (box.alignTop ? 0 : (bh - h) / 2);
  slide.addImage({ path: path.join(CROPS, name + ".png"), x, y, w, h, altText: box.alt || name });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background1, transparency: 100 },
    line: { color: C.text1, width: 0.75 }, objectName: "graphic frame" });
  if (box.caption) {
    slide.addText(box.caption, { x, y: y + h + 0.04, w, h: 0.26, fontSize: 10, color: C.accent2, italic: true,
      margin: 0, align: "right", isTextBox: true });
  }
  return { x, y, w, h };
}

// callouts at fractions of a placed picture
function callouts(slide, rect, pts) {
  pts.forEach(([n, fx, fy]) => badge(slide, n, rect.x + fx * rect.w - 0.18, rect.y + fy * rect.h - 0.18, 0.36, true));
}

function base(title, bottom, section) {
  const slide = pres.addSlide({ masterName: "RIBBONS", sectionTitle: section });
  slide.addText(title, { placeholder: "title" });
  slide.addText(bottom, { placeholder: "bottomline" });
  return slide;
}

const SIM = "Simulated data and network drive";

// generic boxed diagram node
function node(slide, x, y, w, h, title, lines = [], opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opts.fill ?? C.background1 },
    line: { color: C.text1, width: opts.lineW ?? 1 }, objectName: `node ${title}` });
  const runs = [{ text: title, options: { bold: true, fontSize: opts.titleSize ?? 14, color: opts.color ?? C.text1,
    breakLine: lines.length > 0 } }];
  lines.forEach((l, i) => runs.push({ text: l, options: { fontSize: opts.size ?? 12, color: opts.color ?? C.text1,
    breakLine: i < lines.length - 1 } }));
  slide.addText(runs, { x: x + 0.12, y: y + 0.08, w: w - 0.24, h: h - 0.16, valign: opts.valign ?? "top",
    align: opts.align ?? "left", margin: 0, isTextBox: true, paraSpaceAfter: 2 });
}
function arrow(slide, x1, y1, x2, y2, label) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1,
    line: { color: C.text1, width: 1.5, endArrowType: "triangle" } });
  if (label) slide.addText(label, { x: Math.min(x1, x2) - 0.6, y: (y1 + y2) / 2 - 0.42, w: Math.abs(x2 - x1) + 1.2,
    h: 0.32, align: "center", fontSize: 11, color: C.text1, margin: 0, isTextBox: true });
}
function frame(slide, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background1, transparency: 100 },
    line: { color: C.text1, width: 0.75 }, objectName: "graphic frame" });
}

// ============================ 1. Title ===================================================================
pres.addSection({ title: "Introduction" });
{
  const s = base("MicroSegWorkbench: a beginner's guide", "Six guided steps; the app checks your work at every step",
    "Introduction");
  s.addText([
    { text: "Train, check, compare and share microstructure segmentation models", options: { bold: true,
      fontSize: 22, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "For researchers with no programming background", options: { fontSize: 16, breakLine: true } },
    { text: `Version ${VERSION} · November 2026`, options: { fontSize: 16 } },
  ], { x: LEFT_X, y: BODY_Y + 0.3, w: LEFT_W, h: 2.6, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addText([
    { text: "About the screenshots", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "Real app, real training on CPU. Images are synthetic; a local folder (/mnt/office-share) stands in for the office network drive.", options: { fontSize: 14 } },
  ], { x: LEFT_X, y: 4.9, w: LEFT_W, h: 1.4, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  picture(s, "full_data_ready", { caption: SIM });
  s.addNotes("Use this deck next to the in-app Help page. Each slide is one step; numbered markers on the screenshot match the numbered statements on the left. Boxes marked ATTENTION are the points where beginners most often go wrong.");
}

// ============================ 2. What it does =========================================================
{
  const s = base("What the workbench does", "One guided path from annotated micrographs to an approved model",
    "Introduction");
  statements(s, [
    { head: "Trains segmentation networks", text: "on your annotated micrographs, on a GPU or the CPU" },
    { head: "Keeps every result traceable", text: "to exact images, settings, network and software" },
    { head: "Checks your work at each step", text: "data, settings, training and results, in plain words" },
    { head: "Shares through the office drive", text: "a colleague reviews before a model counts as approved" },
  ]);
  // flow diagram: 8 stages in two rows, grouped by app step
  const gx = RIGHT_X + 0.25, gw = RIGHT_W - 0.5;
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  const stages = [
    ["1", "Images & masks", "check folder"], ["1", "Dataset version", "frozen split"],
    ["2", "Preprocess", "border, size"], ["2", "Augment", "visual pipeline"],
    ["3", "Train", "live curves"], ["4", "Evaluate", "test scores, notes"],
    ["5", "Compare", "runs, same image"], ["6", "Share", "publish, review"],
  ];
  const bw = 1.55, bh = 1.05, colGap = (gw - 4 * bw) / 3;
  const rows = [BODY_Y + 0.75, BODY_Y + 3.05];
  stages.forEach(([step, t, sub], i) => {
    const r = Math.floor(i / 4), c = r === 0 ? i % 4 : 3 - (i % 4);
    const x = gx + c * (bw + colGap), y = rows[r];
    node(s, x, y, bw, bh, t, [sub], { align: "center", valign: "middle", titleSize: 13, size: 11,
      fill: step % 2 ? C.background1 : C.background2 });
    badge(s, step, x - 0.15, y - 0.15, 0.32);
    if (i < 7) {
      if (i === 3) arrow(s, x + bw / 2, y + bh, x + bw / 2, rows[1]);
      else if (r === 0) arrow(s, x + bw, y + bh / 2, x + bw + colGap, y + bh / 2);
      else arrow(s, x, y + bh / 2, x - colGap, y + bh / 2);
    }
  });
  s.addText("Numbers = steps in the app's left-hand list", { x: gx, y: BODY_B - 0.85, w: gw, h: 0.3, fontSize: 12,
    color: C.text1, italic: true, align: "center", margin: 0, isTextBox: true });
  s.addText("Help (?) explains every step", { x: gx, y: BODY_B - 0.55, w: gw, h: 0.3, fontSize: 12,
    color: C.text1, italic: true, align: "center", margin: 0, isTextBox: true });
}

// ============================ 3. Where files live =====================================================
{
  const s = base("Where your files live", "Originals stay on the network drive; your PC works on verified copies",
    "Introduction");
  statements(s, [
    { head: "Network drive: the authoritative copy", text: "micrographs, dataset versions, published models, reviews" },
    { head: "Your PC: the workspace", text: "verified local copies, runs in progress, caches" },
    { head: "Publishing is checked", text: "files are fingerprinted, never overwritten, never half-written" },
  ]);
  attention(s, "Never move, rename or edit files inside the workspace or the shared folder by hand. Use the app.");
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  const nx = RIGHT_X + 0.35;
  node(s, nx, BODY_Y + 0.35, 3.0, 2.2, "Office network drive", ["(S:\\ or \\\\server\\share)", " ",
    "micrographs and masks", "MicroSegWorkbench shared folder:", "  dataset versions", "  published models + notes",
    "  reviews"], { fill: C.background2 });
  node(s, nx + 4.0, BODY_Y + 0.35, 2.9, 2.2, "Your Windows PC", ["local workspace", " ", "working copies (verified)",
    "training runs, checkpoints", "predictions", "uses NVIDIA GPU or CPU"]);
  node(s, nx + 4.0, BODY_Y + 3.35, 2.9, 1.55, "Ubuntu VM (optional)", ["small index for search and", "review status; no images"]);
  arrow(s, nx + 3.0, BODY_Y + 1.15, nx + 4.0, BODY_Y + 1.15, "download");
  arrow(s, nx + 4.0, BODY_Y + 1.75, nx + 3.0, BODY_Y + 1.75, "publish");
  arrow(s, nx + 5.45, BODY_Y + 2.55, nx + 5.45, BODY_Y + 3.35);
  node(s, nx, BODY_Y + 3.35, 3.0, 1.55, "In these screenshots", ["/mnt/office-share = network drive", "alice, bob, carol, dave =",
    "four colleagues' PCs (simulated)"], { lineW: 0.75 });
}

// ============================ 3b. Installation check ===================================================
{
  const s = base("Check the installation first", "A READY self-test on every workstation before real work",
    "Introduction");
  statements(s, [
    { n: 1, head: "Help -> Check this installation", text: "or msw selftest; about a minute" },
    { n: 2, head: "Real training on tiny synthetic data", text: "GPU, disk, data, training, evaluation, segmentation, reproducibility" },
    { n: 3, head: "Shared folder: read check", text: "nothing is written; IT tests publishing once per share (--share-write)" },
    { head: "Report saved for sign-off", text: "workspace\\selftest-reports" },
  ]);
  attention(s, "NOT READY? Each FAIL line names the step and the reason; send the report to IT.");
  const r = picture(s, "selftest", { caption: "CPU-only sandbox: the GPU check shows WARN" });
  callouts(s, r, NUM.callouts.selftest);
}

// ============================ 4. Screen tour ==========================================================
{
  const s = base("Finding your way around", "Follow the map left to right; the footer always says what to do next",
    "Introduction");
  statements(s, [
    { n: 1, head: "Steps 1-6 on the left", text: "a tick when done, ! when something needs a look" },
    { n: 2, head: "Workflow map on top", text: "click a stage to jump there" },
    { n: 3, head: "Blue box explains the step" },
    { n: 4, head: "Footer: what to do here", text: "plus Back and Next" },
    { n: 5, head: "Beginner / Advanced mode", text: "Beginner shows only what you need" },
  ]);
  const r = picture(s, "tour", { caption: SIM });
  callouts(s, r, NUM.callouts.tour);
}

// ============================ 5. Prepare folder =======================================================
pres.addSection({ title: "Step 1 - Data" });
{
  const s = base("Step 1 · Prepare the image folder", "Correct names and exact mask values are the foundation of every result",
    "Step 1 - Data");
  statements(s, [
    { head: "One mask per image", text: "S01_0001.png with S01_0001_mask.png" },
    { head: "Mask pixel value = class number", text: "classes.json names every value" },
    { head: "Name starts with the specimen", text: "S01_... keeps a specimen's images together" },
    { head: "Or import an OnlineAnnotator export", text: "approved annotations, provenance kept" },
  ]);
  attention(s, "Save masks as PNG. JPEG and resizing change pixel values and silently create wrong classes.");
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  const fx = RIGHT_X + 0.4;
  s.addText("Zr-hydride-batch-01\\", { x: fx, y: BODY_Y + 0.3, w: 3.4, h: 0.35, bold: true, fontSize: 15,
    fontFace: "Courier New", color: C.text1, margin: 0, isTextBox: true });
  const files = ["classes.json", "S01_0001.png", "S01_0001_mask.png", "S01_0002.png", "S01_0002_mask.png", "...",
    "S06_0004.png", "S06_0004_mask.png"];
  files.forEach((f, i) => s.addText(f, { x: fx + 0.35, y: BODY_Y + 0.75 + i * 0.36, w: 3.2, h: 0.32,
    fontSize: 14, fontFace: "Courier New", color: C.text1, margin: 0, isTextBox: true }));
  // classes.json box
  node(s, fx + 3.85, BODY_Y + 0.3, 2.85, 1.65, "classes.json", [" ", "0 = matrix", "1 = hydride", "2 = pore"],
    { fill: C.background2, size: 14 });
  // naming explanation
  s.addShape(pres.shapes.RECTANGLE, { x: fx + 3.85, y: BODY_Y + 2.3, w: 2.85, h: 2.6, fill: { color: C.background1 },
    line: { color: C.text1, width: 1 } });
  s.addText([
    { text: "S01", options: { bold: true, fontFace: "Courier New", fontSize: 20, breakLine: false } },
    { text: "_0001", options: { fontFace: "Courier New", fontSize: 20, breakLine: true } },
    { text: "^ specimen (group)", options: { fontFace: "Courier New", fontSize: 12, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Group pattern ^([^_]+)_ takes the text before the first _ as the specimen.", options: { fontSize: 13,
      breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "All images of one specimen go to the same split.", options: { fontSize: 13 } },
  ], { x: fx + 4.0, y: BODY_Y + 2.42, w: 2.6, h: 2.4, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  // mask pixel sketch
  const gx0 = fx + 0.4, gy0 = BODY_Y + 3.85, cell = 0.22;
  const grid = ["000110", "001100", "011000", "110020", "100222", "000020"];
  grid.forEach((row, r) => [...row].forEach((v, c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: gx0 + c * cell, y: gy0 + r * cell, w: cell, h: cell,
      fill: { color: v === "0" ? "FFFFFF" : v === "1" ? "7F7F7F" : "262626" }, line: { color: "BFBFBF", width: 0.5 } });
    s.addText(v, { x: gx0 + c * cell, y: gy0 + r * cell, w: cell, h: cell, fontSize: 9, align: "center",
      valign: "middle", color: v === "2" ? "FFFFFF" : "000000", margin: 0, isTextBox: true });
  }));
  s.addText("A mask is an image of class numbers, one per pixel", { x: gx0 + 1.5, y: gy0 + 0.3, w: 1.9, h: 0.9,
    fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
}

// ============================ 6. Check folder ==========================================================
{
  const s = base("Step 1 · Check the folder", "Fix every red problem; the app will not build a dataset from a broken folder",
    "Step 1 - Data");
  statements(s, [
    { n: 1, head: "Browse to the folder, press Check folder" },
    { n: 2, head: "Errors listed in plain words", text: "here: a missing mask and an undeclared value 7" },
    { n: 3, head: "Problem pictures framed red", text: "click one to see it" },
    { n: 4, head: "Class balance", text: "rare classes flagged in amber" },
  ]);
  attention(s, "Fix problems in the original files, then press Check folder again.");
  const r = picture(s, "data_problems", { caption: SIM });
  callouts(s, r, NUM.callouts.data_problems);
}

// ============================ 7. Split ==========================================================
{
  const s = base("Step 1 · Split by specimen", "Splitting by specimen keeps the test honest", "Step 1 - Data");
  statements(s, [
    { n: 1, head: "Image and mask side by side", text: "click any picture in the gallery" },
    { n: 2, head: "Mask opacity slider", text: "check that masks sit exactly on the features" },
    { n: 3, head: "Whole specimens per split", text: "Training learns, Validation picks the best epoch, Test is scored once" },
  ]);
  attention(s, "With one test specimen, test scores describe only that specimen. Use more specimens where you can.");
  const r = picture(s, "data_ready", { caption: SIM });
  callouts(s, r, NUM.callouts.data_ready);
}

// ============================ 8. Dataset version ========================================================
{
  const s = base("Step 1 · Create the dataset version", "A dataset version is frozen: every result points to exactly these files",
    "Step 1 - Data");
  statements(s, [
    { n: 1, head: "Give the version a name" },
    { n: 2, head: "Create dataset version", text: "a frozen copy with fingerprints; the same version always gives the same split" },
    { n: 3, head: "Dataset check runs automatically", text: "green tick, blue note, amber warning or red block" },
  ]);
  const r = picture(s, "dataset_version", { caption: SIM });
  callouts(s, r, NUM.callouts.dataset_version);
  attention(s, "Adding or fixing images later? Create a new version; old results keep pointing to the old one.");
}

// ============================ 9. Dataset check block ===================================================
{
  const s = base("When the dataset check stops you", "A blocked start saves hours of training a useless model",
    "Step 1 - Data");
  statements(s, [
    { n: 1, head: "Class missing from training", text: "pores occur only in specimen S03, which went to the test split" },
    { n: 2, head: "Start training stays disabled", text: "the checklist says why" },
    { head: "Fix the data, not the check", text: "annotate pores in more specimens, then create a new version" },
  ]);
  attention(s, "Amber warnings need a look; red blocks need a fix before training.");
  const r1 = picture(s, "health_data", { y: BODY_Y, h: 1.25, caption: null });
  const r2 = picture(s, "health_train", { y: BODY_Y + 1.45, h: 3.85, caption: "Simulated example: dave's dataset" });
  callouts(s, r1, NUM.callouts.health_data);
  callouts(s, r2, NUM.callouts.health_train);
}

// ============================ 10. Preprocessing ========================================================
pres.addSection({ title: "Step 2 - Preprocess and augment" });
{
  const s = base("Step 2A · Preprocessing", "Preprocess as little as possible; never shrink features away",
    "Step 2 - Preprocess and augment");
  statements(s, [
    { n: 1, head: "Drag the dashed edges", text: "to leave out data bars or labels; excluded pixels are never trained or scored" },
    { n: 2, head: "Resize only for very large images", text: "before/after at true pixel size" },
    { n: 3, head: "Feature-width warning", text: "hydrides 3 px wide become 0.7 px at x0.25" },
  ]);
  attention(s, "Keep resize at x1 unless memory forces it. Features thinner than ~1.5 px break apart.");
  const r = picture(s, "preprocess", { caption: SIM });
  callouts(s, r, NUM.callouts.preprocess);
}

// ============================ 11. Augmentation builder ================================================
{
  const s = base("Step 2B · The augmentation pipeline", "Start from a preset; change a block only for a reason you can name",
    "Step 2 - Preprocess and augment");
  statements(s, [
    { n: 1, head: "Start from a preset", text: "Standard suits most data" },
    { n: 2, head: "Blocks run left to right", text: "blue: image and mask move together; orange: image only" },
    { n: 3, head: "Dots = how often a block fires", text: "slider sets probability, thumbnails show weakest and strongest" },
    { n: 4, head: "Real training patches", text: "exactly what the model will see" },
  ]);
  const r = picture(s, "augment_builder", { caption: SIM });
  callouts(s, r, NUM.callouts.augment_builder);
}

// ============================ 12. Step by step + check =================================================
{
  const s = base("Step 2B · Check before you train", "Run 'Check these settings' after every change",
    "Step 2 - Preprocess and augment");
  statements(s, [
    { n: 1, head: "Step by step", text: "one patch followed through every block" },
    { n: 2, head: "Check these settings", text: "runs the real pipeline on your images" },
    { n: 3, head: "Findings in plain words", text: "clipping, ignored pixels, blur against feature width, fire rates" },
  ]);
  attention(s, "Strong blur or contrast changes can erase thin features. Read every finding.");
  const r1 = picture(s, "augment_steps", { y: BODY_Y, h: 2.15, caption: null });
  const r2 = picture(s, "augment_check", { y: BODY_Y + 2.35, h: 2.95, caption: SIM });
  callouts(s, r1, NUM.callouts.augment_steps);
  callouts(s, r2, NUM.callouts.augment_check);
}

// ============================ 13. Train setup ==========================================================
pres.addSection({ title: "Step 3 - Train" });
{
  const s = base("Step 3 · Set up a training run", "Read the checklist and the run card before pressing Start",
    "Step 3 - Train");
  statements(s, [
    { n: 1, head: "Dataset version and network" },
    { n: 2, head: "Preset", text: "Quick CPU for a first try; Standard or Large GPU for real runs" },
    { n: 3, head: "Ready to train?", text: "all ticks, or amber items you have read" },
    { n: 4, head: "This run", text: "epochs, model size, time and memory, measured on this PC" },
  ]);
  attention(s, "No NVIDIA GPU found means training runs on the CPU: slower, but it works.");
  const r = picture(s, "train_setup", { caption: SIM });
  callouts(s, r, NUM.callouts.train_setup);
}

// ============================ 14. Choosing a network ===================================================
{
  const s = base("Choosing a network", "Start with U-Net; try the transformer when you have more data", "Step 3 - Train");
  statements(s, [
    { head: "U-Net: first choice", text: "convolutional, fast, reliable on small datasets" },
    { head: "SegFormer-lite: transformer", text: "sees wider context; needs more images and epochs" },
    { head: "Train both on the same dataset version", text: "then compare them in step 5" },
    { head: "More networks can be added later", text: "as plugins, without changing the app" },
  ]);
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  const rows = [
    ["", "U-Net", "SegFormer-lite"],
    ["Type", "Convolutional encoder-decoder", "Vision transformer, MLP decoder"],
    ["Strength", "Fine detail, small data", "Long-range context"],
    ["Needs", "Few images, short training", "More images, more epochs"],
    ["Patch size", "multiple of 2^depth", "multiple of 32"],
    ["Parameters (quick preset)", NUM.unet_params, NUM.seg_params],
    ["Demo test feature IoU", NUM.unet_iou, NUM.seg_iou],
    ["Epochs in the demo", String(NUM.unet_epochs), String(NUM.seg_epochs)],
  ];
  const tableRows = rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    bold: i === 0 || j === 0, fontSize: 14, color: i === 0 ? C.background1 : C.text1,
    fill: { color: i === 0 ? C.text1 : (i % 2 ? C.background1 : C.background2) }, valign: "middle",
    align: j === 0 ? "left" : "center" } })));
  s.addTable(tableRows, { x: RIGHT_X + 0.3, y: BODY_Y + 0.3, w: RIGHT_W - 0.6, colW: [2.2, 2.2, 2.43],
    rowH: 0.5, border: { type: "solid", color: "7F7F7F", pt: 0.5 }, fontFace: "Arial", margin: 0.06 });
  s.addText(`Demo: synthetic data, 16 training images, CPU; U-Net ${NUM.unet_epochs} epochs, SegFormer-lite ${NUM.seg_epochs} epochs. Your numbers will differ.`,
    { x: RIGHT_X + 0.3, y: BODY_B - 0.85, w: RIGHT_W - 0.6, h: 0.6, fontSize: 12, italic: true, color: C.text1,
      margin: 0, isTextBox: true });
}

// ============================ 15. Beginner / Advanced ===================================================
{
  const s = base("Beginner and Advanced mode", "Stay in Beginner mode until a result tells you what to change",
    "Step 3 - Train");
  statements(s, [
    { head: "Switch at the top right", text: "remembered for your workspace" },
    { n: 2, head: "Advanced adds expert settings", text: "batch, learning rate, loss, seed, repeats, precision, early stopping" },
    { head: "Beginner keeps the essentials", text: "dataset, network, preset, epochs, device" },
  ]);
  attention(s, "Change one expert setting at a time, and compare the runs in step 5.");
  const r = picture(s, "train_advanced", { caption: SIM });
  callouts(s, r, NUM.callouts.train_advanced);
}

// ============================ 16. Live training ========================================================
{
  const s = base("Step 3 · Watching training", "Judge training by validation curves and images, not by loss alone",
    "Step 3 - Train");
  statements(s, [
    { n: 1, head: "Progress and time left" },
    { n: 2, head: "Loss falls", text: "training (blue) and validation (orange)" },
    { n: 3, head: "Validation IoU rises, then levels off", text: "the best epoch is kept automatically" },
    { n: 4, head: "Prediction evolution", text: "the same image, epoch by epoch" },
  ]);
  attention(s, "Training loss falling while validation IoU drops = overfitting. Cancel is safe; resume later in step 4.");
  const r = picture(s, "train_live", { caption: SIM });
  callouts(s, r, NUM.callouts.train_live);
}

// ============================ 17. Resources ==========================================================
{
  const s = base("Step 3 · Computer resources", "Know whether you are on the GPU or the CPU before a long run",
    "Step 3 - Train");
  statements(s, [
    { n: 1, head: "Memory of the run and the PC", text: "and GPU memory when an NVIDIA GPU is used" },
    { n: 2, head: "Measured speed" },
    { n: 3, head: "Utilisation over time" },
    { head: "Out of memory?", text: "the run stops cleanly and suggests a smaller batch or patch" },
  ]);
  const r = picture(s, "train_resources", { caption: "CPU-only sandbox: GPU memory not available" });
  callouts(s, r, NUM.callouts.train_resources);
}

// ============================ 18. Results ==========================================================
pres.addSection({ title: "Step 4 - Results" });
{
  const s = base("Step 4 · Results and run check", "Results stay unreviewed until a colleague approves them",
    "Step 4 - Results");
  statements(s, [
    { n: 1, head: "Status banner: UNREVIEWED" },
    { n: 2, head: "Run check in plain words", text: "overfitting, under-training, missed classes, thin evidence" },
    { n: 3, head: "Per-class test scores", text: "on the test split, never used for training" },
  ]);
  attention(s, "Judge by the feature classes (hydride, pore). Matrix dominates the pixels, so overall accuracy flatters.");
  const r = picture(s, "results", { caption: SIM });
  callouts(s, r, NUM.callouts.results);
}

// ============================ 19. Overlays ==========================================================
{
  const s = base("Step 4 · Look at the overlays", "A good score can hide systematic errors; always look", "Step 4 - Results");
  statements(s, [
    { n: 1, head: "Every test image", text: "ground truth next to prediction" },
    { head: "Look for", text: "broken thin hydrides, missed or merged pores, false edges at borders" },
    { head: "Export report", text: "one self-contained HTML file to attach or archive" },
  ]);
  const r = picture(s, "results_overlays", { caption: SIM });
  callouts(s, r, NUM.callouts.results_overlays);
}

// ============================ 19b. Accuracy details ==================================================
{
  const s = base("Step 4 · Accuracy details", "Check boundaries and area fractions before you measure with a model",
    "Step 4 - Results");
  statements(s, [
    { n: 1, head: "Boundary F1", text: "share of edges within 1 px of the true edge; 1 = every edge in place" },
    { n: 2, head: "True vs predicted area %", text: "a clear difference means biased phase fractions" },
    { n: 3, head: "Weakest test images", text: "look at their overlays first" },
    { n: 4, head: "Where pixels go", text: "which classes are confused with which" },
  ]);
  attention(s, "Before reporting area fractions, compare them with manual measurements on images not used for training.");
  const r = picture(s, "results_accuracy", { caption: SIM });
  callouts(s, r, NUM.callouts.results_accuracy);
}

// ============================ 20. Observations & card ================================================
{
  const s = base("Step 4 · Observations and model card", "Write down what you checked; colleagues rely on it",
    "Step 4 - Results");
  statements(s, [
    { n: 1, head: "Observations", text: "what you checked, what the model is and is not good for" },
    { head: "Notes cannot be edited", text: "add a new note to correct one" },
    { n: 2, head: "Model card", text: "network, classes, input, training data, scores, software, checkpoint fingerprint" },
  ]);
  attention(s, "Publishing in step 6 needs at least one observation.");
  const r1 = picture(s, "results_observations", { y: BODY_Y, h: 2.55, caption: null });
  const r2 = picture(s, "results_card", { y: BODY_Y + 2.7, h: 2.6, caption: SIM });
  callouts(s, r1, NUM.callouts.results_observations);
  callouts(s, r2, NUM.callouts.results_card);
}

// ============================ 21. Compare ==========================================================
pres.addSection({ title: "Step 5 - Compare" });
{
  const s = base("Step 5 · Compare runs", "Compare like with like: same dataset version, several seeds", "Step 5 - Compare");
  statements(s, [
    { n: 1, head: "Tick the runs to compare" },
    { n: 2, head: "Scores and settings side by side", text: "'What differs' lists only the changed settings" },
    { n: 3, head: "Validation curves, test predictions" },
  ]);
  attention(s, "Rank runs only on the same dataset version. Repeat with seeds (Advanced) to see how much is luck.");
  const r = picture(s, "compare_scores", { caption: SIM });
  callouts(s, r, NUM.callouts.compare_scores);
}

// ============================ 22. Same image ==========================================================
{
  const s = base("Step 5 · Same image, several models", "Disagreement shows where to look more closely",
    "Step 5 - Compare");
  statements(s, [
    { n: 1, head: "Choose a dataset image or any file", text: "dataset images bring their ground truth" },
    { n: 2, head: "Each model's segmentation", text: "U-Net and SegFormer-lite side by side" },
    { n: 3, head: "Where models disagree", text: "orange pixels" },
    { n: 4, head: "Scores, class fractions, time" },
  ]);
  const r = picture(s, "compare_same_image", { caption: SIM });
  callouts(s, r, NUM.callouts.compare_same_image);
}

// ============================ 23. Publish ==========================================================
pres.addSection({ title: "Step 6 - Share" });
{
  const s = base("Step 6 · Connect and publish", "Publishing makes your model available, with your notes attached",
    "Step 6 - Share");
  statements(s, [
    { n: 1, head: "Shared folder path and your name", text: "once; Save and connect" },
    { n: 2, head: "Publish the dataset version first" },
    { n: 3, head: "Write observations, publish the run", text: "files verified, deduplicated, immutable" },
  ]);
  attention(s, "Published items are never overwritten. To improve a model, train and publish a new run.");
  const r1 = picture(s, "share_connect", { y: BODY_Y, h: 1.55, caption: null });
  const r2 = picture(s, "share_runpub", { y: BODY_Y + 1.7, h: 3.6, caption: SIM });
  callouts(s, r1, NUM.callouts.share_connect);
  callouts(s, r2, NUM.callouts.share_runpub);
}

// ============================ 24. Review ==========================================================
{
  const s = base("Step 6 · Review by a colleague", "Approval must come from someone other than the publisher",
    "Step 6 - Share");
  statements(s, [
    { head: "Download the run", text: "with its dataset, verified" },
    { n: 2, head: "Read the observations, check the overlays" },
    { n: 3, head: "Decide and comment", text: "approved, needs changes or rejected" },
    { n: 4, head: "Status for everyone", text: "UNREVIEWED becomes APPROVED" },
  ]);
  const r1 = picture(s, "share_review_table", { y: BODY_Y, h: 0.6, caption: null });
  const r2 = picture(s, "share_review_panel", { y: BODY_Y + 0.8, h: 4.5, caption: SIM + " (bob's PC)" });
  callouts(s, r1, NUM.callouts.share_review_table);
  callouts(s, r2, NUM.callouts.share_review_panel);
}

// ============================ 25. Fetch and use ========================================================
{
  const s = base("Fetch and use an approved model", "Heed input warnings: a model knows only images like its training images",
    "Step 6 - Share");
  statements(s, [
    { n: 1, head: "Approved models only" },
    { n: 2, head: "Download and segment images", text: "pick a folder; model and data are fetched, then applied" },
    { n: 3, head: "Warning for unlike images", text: "here: an over-exposed micrograph, on which the model found no hydrides at all" },
  ]);
  attention(s, "Predictions are unreviewed. Check images with warnings, or retrain with such images.");
  const r1 = picture(s, "share_fetch", { y: BODY_Y, h: 1.95, caption: null });
  const r2 = picture(s, "carol_results", { y: BODY_Y + 2.15, h: 3.15,
    caption: "Warning, then the output overlay: image | prediction (nothing found)" });
  callouts(s, r1, NUM.callouts.share_fetch);
  callouts(s, r2, NUM.callouts.carol_results);
}

// ============================ 26. CLI ==========================================================
pres.addSection({ title: "Reference" });
{
  const s = base("The same work from the command line", "Everything in the app is also available as msw commands",
    "Reference");
  statements(s, [
    { head: "For repeat or batch work", text: "same checks, same files, same provenance" },
    { head: "msw models", text: "every model here and on the shared folder" },
    { head: "msw predict", text: "downloads a shared model first" },
    { head: "msw selftest", text: "checks this computer and the shared folder" },
    { head: "msw --help", text: "lists every command" },
  ]);
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  s.addShape(pres.shapes.RECTANGLE, { x: RIGHT_X + 0.2, y: BODY_Y + 0.2, w: RIGHT_W - 0.4, h: BODY_B - BODY_Y - 0.4,
    fill: { color: C.text1 }, line: { color: C.text1, width: 0 } });
  s.addText(NUM.cli.map((l, i) => ({ text: l, options: { breakLine: i < NUM.cli.length - 1,
    bold: l.startsWith(">"), color: l.startsWith(">") ? C.background1 : "D9D9D9" } })),
  { x: RIGHT_X + 0.4, y: BODY_Y + 0.35, w: RIGHT_W - 0.8, h: BODY_B - BODY_Y - 0.7, fontFace: "Courier New",
    fontSize: 11, valign: "top", margin: 0, isTextBox: true });
}

// ============================ 27. Checklist ==========================================================
{
  const s = base("Ten things that need your attention", "Most mistakes are caught by the app; the rest by your eyes",
    "Reference");
  statements(s, [
    { head: "Before you trust a result" },
    { head: "Before you share it" },
    { head: "Before you use someone else's model" },
    { head: "Print this page", text: "and keep it next to the workstation" },
  ]);
  frame(s, RIGHT_X, BODY_Y, RIGHT_W, BODY_B - BODY_Y);
  const items = [
    "Masks are PNG with exact class values; classes.json names every value",
    "File names start with the specimen, so splits keep specimens together",
    "Fix every red problem; read every amber warning",
    "Never edit files inside the workspace or shared folder by hand",
    "Do not resize features below ~1.5 px",
    "Run 'Check these settings' after changing augmentation",
    "Judge by feature IoU and the overlays, not overall accuracy",
    "Compare runs only on the same dataset version (and several seeds)",
    "Write observations before publishing; never approve your own run",
    "Unreviewed is not approved; check images that raise input warnings",
  ];
  const y0 = BODY_Y + 0.25, rowH = 0.49;
  items.forEach((t, i) => {
    badge(s, i + 1, RIGHT_X + 0.3, y0 + i * rowH + 0.04, 0.34);
    s.addText(t, { x: RIGHT_X + 0.85, y: y0 + i * rowH, w: RIGHT_W - 1.1, h: 0.42, fontSize: 14, valign: "middle",
      color: C.text1, margin: 0, isTextBox: true });
  });
}

// ============================ 28. Help ==========================================================
{
  const s = base("Getting help", "When unsure: read the step's help, then ask a colleague", "Reference");
  statements(s, [
    { n: 1, head: "? Help page", text: "every step explained, with the matching commands" },
    { head: "Hover for tooltips", text: "on every button and setting" },
    { head: "Error messages say what to do", text: "not just what went wrong" },
    { head: "Administrators", text: "README, docs/DEPLOYMENT.md, docs/SPECIFICATION.md" },
  ]);
  const r = picture(s, "help", { caption: SIM });
  callouts(s, r, NUM.callouts.help);
}

(async () => {
  const out = path.join(BUILD, "MicroSegWorkbench_Beginners_Guide.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
})();
