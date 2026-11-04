# Beginner's guide (slides) - how it is built

The published guide is `docs/guide/MicroSegWorkbench_Beginners_Guide.pdf` (and `.pptx`). It is generated from the
real application so the screenshots match the software:

1. `python shoot.py` - drives the desktop app offscreen through a simulated office day on SYNTHETIC data (alice
   trains U-Net and SegFormer-lite and publishes, bob reviews, carol downloads and segments new images, dave's
   dataset is blocked) and saves screenshots. Training is real (CPU, about 10 minutes). The simulated network drive
   and workstations default to `/mnt/office-share` and `/srv/workstations` (override with `MSW_GUIDE_SHARE`,
   `MSW_GUIDE_HOMES`; both are deleted and recreated). Needs PySide6 (offscreen) and the package's dependencies.
2. `python crops.py` - crops the screenshots (boxes in the script) and writes `grid/` previews with a 10 % grid for
   placing the numbered markers, whose positions are in `numbers.json`.
3. `node deck.js` - builds the slides with `pptxgenjs` (`npm install pptxgenjs@3.12.0`), house style: black title
   and bottom-line ribbons, Arial, short statements left, graphic right.
4. Convert to PDF with PowerPoint or LibreOffice and copy both files to `docs/guide/`.

Everything goes to `build/` (or `MSW_GUIDE_BUILD`), which is not committed. After UI changes, re-run all steps and
check the marker positions in the rendered slides.
