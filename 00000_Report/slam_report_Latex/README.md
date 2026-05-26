# English SLAM Course Report

This directory contains the English-only version of the SLAM course report.

## Compile

Recommended TeXWorks sequence:

```text
PDFLaTeX
BibTeX8
PDFLaTeX
PDFLaTeX
```

Command-line equivalent:

```bash
pdflatex main.tex
bibtex8 main
pdflatex main.tex
pdflatex main.tex
```

If citations appear as `[?]`, BibTeX8 has not been run or the `.bbl` file has not been regenerated.

## Notes

- The report uses English section titles, captions, tables, and body text.
- The workflow figure is loaded from `figures/Workflow.png`.
- The report keeps the claim that the implementation is a Python research baseline, not an official full reproduction of ORB-SLAM2, DSO, or SVO.
- The project manifest claim remains `paper_level_claim=false`.
