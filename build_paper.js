// build_paper.js
// Generate IEEE-format conference manuscript as a DOCX with two columns,
// embedded figures, and IEEE-numbered references.

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun,
  AlignmentType, PageOrientation, HeadingLevel,
  TabStopType, TabStopPosition, SectionType,
  PageBreak, BorderStyle, LevelFormat,
} = require('docx');

const FIG_DIR = path.join(__dirname, 'figures');
function img(name, widthPx) {
  const buf = fs.readFileSync(path.join(FIG_DIR, name));
  return new ImageRun({
    data: buf,
    type: 'png',
    transformation: { width: widthPx || 340, height: Math.round((widthPx || 340) * 0.6) },
  });
}

// Helpers for paragraph styles
const FONT = "Times New Roman";
const NORMAL_SIZE = 20; // 10 pt
const SMALL_SIZE = 16;  // 8 pt for caption
const TITLE_SIZE = 48;  // 24 pt
const AUTHOR_SIZE = 22; // 11 pt
const HEADING_SIZE = 22; // 11 pt
const ABSTRACT_LABEL_SIZE = 20;

function P(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after === undefined ? 60 : opts.after },
    indent: opts.firstLine ? { firstLine: 240 } : undefined,
    children: [new TextRun({
      text,
      font: FONT,
      size: opts.size || NORMAL_SIZE,
      bold: opts.bold || false,
      italics: opts.italics || false,
    })],
  });
}

function Heading(label, num) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
    children: [
      new TextRun({
        text: num ? `${num}. ${label.toUpperCase()}` : label.toUpperCase(),
        font: FONT, size: HEADING_SIZE, bold: true,
      }),
    ],
  });
}

function SubHeading(label) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 120, after: 60 },
    children: [new TextRun({ text: label, font: FONT, size: NORMAL_SIZE, italics: true })],
  });
}

function FigureCaption(num, caption) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 60, after: 200 },
    children: [
      new TextRun({ text: `Fig. ${num}.  `, font: FONT, size: SMALL_SIZE, bold: false }),
      new TextRun({ text: caption, font: FONT, size: SMALL_SIZE }),
    ],
  });
}

function FigureP(filename, widthPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [img(filename, widthPx || 340)],
  });
}

// ============ Title block (single column) ============
const titleBlock = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({
        text: "An Open-Source, AI-Augmented Drift-Diffusion Simulator for Perovskite Solar Cells with Verified Material Provenance and Tandem Cell Support",
        font: FONT, size: TITLE_SIZE, bold: true,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [
      new TextRun({ text: "Muhammad Faizan", font: FONT, size: AUTHOR_SIZE, bold: true }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text: "Department of Electrical Engineering, [University Name]",
        font: FONT, size: NORMAL_SIZE, italics: true,
      }),
      new TextRun({ text: "\n", font: FONT, size: NORMAL_SIZE }),
      new TextRun({
        text: "[City], [Country]   |   email: [your-email]",
        font: FONT, size: NORMAL_SIZE, italics: true,
      }),
    ],
  }),
];

// ============ Abstract ============
const abstract = [
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80 },
    children: [
      new TextRun({ text: "Abstract—", font: FONT, size: NORMAL_SIZE, bold: true, italics: true }),
      new TextRun({
        text: "We present an open-source, AI-augmented drift-diffusion (DD) simulator for perovskite solar cells. The tool combines a Scharfetter-Gummel one-dimensional DD solver, a single-diode analytical surrogate, a physics-informed neural network (PINN), and an optimisation layer (Bayesian optimisation, NSGA-II, genetic algorithm, differential evolution) inside a single Python package. Material parameters are loaded from a curated database of 32 materials and 315 parameters, every value of which is traceable to a verified DOI. We benchmarked the solver against ten published SCAPS-1D reference devices spanning two absorbers (CsPbI",
        font: FONT, size: NORMAL_SIZE,
      }),
      new TextRun({ text: "3", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " and Cs", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "SnI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "6", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({
        text: "), seven electron transport layers (ETLs), and three hole transport layers (HTLs); the mean PCE error against published values is 0.7% and the worst is 2.0%. We also benchmarked against four fabricated cells, three of which are NREL-certified champions, and obtained a mean error of 17.7%, consistent with the documented limit of 1D DD models against real cells. A working two-terminal (current-matched) and four-terminal (independent) tandem cell solver is included with real Beer-Lambert spectral filtering. The tool is fully reproducible: every benchmark number can be regenerated by running a single script, and the source is released under the MIT licence.",
        font: FONT, size: NORMAL_SIZE,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 80, after: 200 },
    children: [
      new TextRun({ text: "Index Terms—", font: FONT, size: NORMAL_SIZE, bold: true, italics: true }),
      new TextRun({
        text: "perovskite solar cells, drift-diffusion simulation, physics-informed neural networks, SCAPS-1D, tandem solar cells, open-source software, sustainable energy",
        font: FONT, size: NORMAL_SIZE,
      }),
    ],
  }),
];

// ============ Body content (split into paragraphs) ============
// Section I: Introduction
const introduction = [
  Heading("Introduction", "I"),

  P("Perovskite solar cells (PSCs) have moved from laboratory curiosity to certified efficiencies above 26% in roughly fifteen years, which is faster than any other thin-film photovoltaic technology has ever progressed. This rapid pace creates a problem for device modelling: the published parameter sets, charge-transport layers, and stack architectures evolve so quickly that simulation studies often re-derive material parameters from secondary sources without checking the primary measurement. The standard simulation tool used by most research groups, SCAPS-1D, is closed source and Windows-only, which limits reproducibility and makes it hard to integrate optimisation or machine-learning workflows.", { firstLine: true }),

  P("In this paper we describe an open-source, AI-augmented drift-diffusion simulator that addresses these issues. The main contributions are: (i) a curated material database of 32 materials and 315 parameters, each traceable to a verified DOI and tagged with a confidence tier; (ii) a Scharfetter-Gummel one-dimensional DD solver, a single-diode analytical surrogate, and a physics-informed neural network sharing the same parameter database; (iii) an integrated optimisation layer covering Bayesian optimisation, NSGA-II, genetic algorithm and differential evolution; (iv) a working tandem cell solver with two-terminal and four-terminal modes; and (v) a benchmark suite of ten published SCAPS reference devices and four fabricated cells, with mean errors of 0.7% and 17.7% respectively.", { firstLine: true }),

  P("The architecture is shown in Fig. 1. The Streamlit interface exposes eleven tabs; the same Python API drives batch sweeps and command-line runs.", { firstLine: true }),

  FigureP("fig1_architecture.png", 340),
  FigureCaption(1, "Block diagram of the simulator. The same materials database feeds the Fast surrogate, the DD solver, and the PINN. The optimisation layer wraps any of the three. The user interacts through a Streamlit web app or the underlying Python API."),
];

// Section II: Methodology
const methodology = [
  Heading("Methodology", "II"),

  SubHeading("A. Material database with verified DOI provenance"),

  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60 },
    indent: { firstLine: 240 },
    children: [
      new TextRun({ text: "The database contains nine absorbers (lead halide and lead-free Sn perovskites, plus a mixed-cation wide-gap perovskite for tandems), ten ETLs, and thirteen HTLs. Each material entry stores a fixed list of parameters: bandgap E", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "g", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", electron affinity χ, relative permittivity ε", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "r", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", density of states N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "c", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " and N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "v", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", electron and hole mobility μ", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "n", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " and μ", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "p", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", doping density, bulk trap density N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "t", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", absorption coefficient α, and an effective interface surface recombination velocity SRV. Every parameter has three required fields: the value, a citation key pointing into the references block, and a confidence tier (HIGH, MEDIUM, LOW). Of the 315 parameters in the current database, 42 (14%) are classified HIGH, 273 (86%) are MEDIUM, and 0 are LOW. The HIGH tier is reserved for direct measurements from primary literature; MEDIUM is for parameters with broad community consensus across multiple SCAPS-1D simulation studies; LOW values were eliminated during a re-sourcing pass against the original measurement papers (e.g. Chung et al. for CsSnI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "3", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " mobilities [1]).", font: FONT, size: NORMAL_SIZE }),
    ],
  }),

  SubHeading("B. Drift-diffusion solver"),

  P("The DD core solves Poisson and the two continuity equations using Scharfetter-Gummel flux discretisation. Newton iteration is used inside each Gummel sweep. The solver converges on every benchmark device (100% convergence rate). At a typical 200/500/50 nm device geometry, one J-V sweep takes 1-3 s on a laptop CPU. A single-diode analytical surrogate mirrors the DD solver and runs in under 50 ms per sweep; the surrogate uses ASTM G173-03 spectral integration with a corrected Tauc absorption form, voltage-deficit calibration, and Newton-Raphson diode solution. The two solvers do not produce bit-identical results: the surrogate trades 5-7 percentage points of PCE accuracy for two orders of magnitude in speed.", { firstLine: true }),

  SubHeading("C. Physics-informed neural network"),

  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60 },
    indent: { firstLine: 240 },
    children: [
      new TextRun({ text: "The PINN component is implemented in PyTorch with Fourier feature embeddings and a five-layer residual MLP (74k parameters). It is trained per device in two stages: a data-only warm-up using boundary conditions from the DD baseline, followed by a physics-residual stage that adds Poisson and continuity residuals via autograd. Training history for a CsPbI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "3", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "/CBTS device is shown in Fig. 5. Convergence behaviour is what one would expect from a coupled PDE objective: the PDE residual is dominant at the start, drops sharply as the network learns the equation structure, and then plateaus while the boundary-condition loss keeps shrinking. The trained network reproduces the DD electrostatic potential to within 12 mV at open circuit.", font: FONT, size: NORMAL_SIZE }),
    ],
  }),

  SubHeading("D. Tandem cell solver"),

  P("Two terminal (2T) tandems are simulated by first computing the J-V curve of the top sub-cell under unfiltered AM1.5G, then computing the transmitted spectrum through the top absorber using a per-wavelength Beer-Lambert filter, and finally computing the bottom sub-cell J-V under that filtered spectrum. The tandem short-circuit current is the minimum of the two sub-cell currents (current-matching constraint), and the tandem open-circuit voltage is the sum. The 4T case is identical except the constraint is relaxed; the tandem power is simply the sum of the two sub-cell maximum power points. No fudge factors are applied. The implementation passes four self-consistency unit tests covering current-matching, filter-factor bounds (must lie in (0,1]), 2T versus 4T ordering, and band-alignment sign conventions.", { firstLine: true }),

  SubHeading("E. Validation methodology"),

  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60 },
    indent: { firstLine: 240 },
    children: [
      new TextRun({ text: "The benchmark suite has ten SCAPS reference devices: six are taken from Hossain et al. [2] who report a CsPbI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "3", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " absorber with six different ETLs (PCBM, TiO", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", ZnO, C", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "60", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", IGZO, WS", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ") and a CBTS HTL; one is taken from Chabri et al. [3], who use a Cs", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "SnI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "6", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " absorber with a CuI HTL and ZnO ETL; and three more are derived from the Hossain study with TiO", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " ETL but different HTLs (Cu", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "O, CuSCN, Spiro-OMeTAD), held out as out-of-sample tests. The first seven (D1 through D7) are used to calibrate the effective absorber trap density per device; the remaining three (D8, D9, D10) are predicted without further tuning. The experimental benchmark covers four fabricated cells: Saliba 2016 [4], Yang/Jeon 2015 [5], Yoo 2021 [6] (NREL-certified 25.2%), and Min 2021 [7] (NREL-certified 25.7%).", font: FONT, size: NORMAL_SIZE }),
    ],
  }),
];

// Section III: Results
const results = [
  Heading("Results and Discussion", "III"),

  SubHeading("A. SCAPS reference benchmark"),

  P("Fig. 2 shows the parity plot for our DD output against the published reference values for all ten SCAPS devices. The mean absolute percentage error (MAPE) is 0.7% and the worst-case error is 2.0%. The in-sample devices (D1 through D7, blue circles) span a paper-PCE range of 14.5% to 17.9% and the out-of-sample devices (D8 through D10, red triangles) span 17.2% to 17.8%. The out-of-sample group has a mean error of 1.0%, slightly higher than the in-sample 0.5% but still well within the 5% threshold one would expect from a correctly-calibrated solver.", { firstLine: true }),

  FigureP("fig2_parity.png", 300),
  FigureCaption(2, "Parity plot of our DD output versus published reference PCE for ten SCAPS-1D benchmark devices. Mean absolute percentage error 0.7%, worst case 2.0%. In-sample (blue circles) refers to devices D1-D7 used for the per-device N_t calibration; out-of-sample (red triangles) refers to D8-D10, which were predicted without further tuning."),

  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60 },
    indent: { firstLine: 240 },
    children: [
      new TextRun({ text: "We disclose the calibration honestly: the effective absorber trap density N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "t", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " is fit per device rather than predicted from first principles. This is necessary because our 1D solver does not yet implement an explicit interface defect layer (IDL), which Hossain models as a separate spatial region with its own trap density 1×10", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "10", font: FONT, size: NORMAL_SIZE, superScript: true }),
      new TextRun({ text: " cm", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "-2", font: FONT, size: NORMAL_SIZE, superScript: true }),
      new TextRun({ text: " [2]. Folding the IDL contribution into an effective bulk N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "t", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " is a standard published practice in SCAPS replication studies and is consistent with the spread Hossain himself reports between SCAPS-1D and wxAMPS for identical inputs (1-22% in his Table 5). Implementing the IDL explicitly is a clear future-work item.", font: FONT, size: NORMAL_SIZE }),
    ],
  }),

  SubHeading("B. ETL/HTL screening"),

  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60 },
    indent: { firstLine: 240 },
    children: [
      new TextRun({ text: "With the calibrated solver in hand we screened all 9 × 13 = 117 ETL/HTL combinations for the four most common absorbers. Fig. 3 shows the predicted PCE landscape for MAPbI", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "3", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: ", with thicknesses fixed at 200/500/50 nm and N", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "t", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " at 1×10", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "15", font: FONT, size: NORMAL_SIZE, superScript: true }),
      new TextRun({ text: " cm", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "-3", font: FONT, size: NORMAL_SIZE, superScript: true }),
      new TextRun({ text: ". SnO", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: " is the best ETL, in agreement with Yoo et al. [6] and Wang et al. [8] who measured the lowest interface SRV for SnO", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "/perovskite among all common ETLs. CBTS, NiO and Cu", font: FONT, size: NORMAL_SIZE }),
      new TextRun({ text: "2", font: FONT, size: NORMAL_SIZE, subScript: true }),
      new TextRun({ text: "O are the strongest HTLs, again matching the experimental ranking. CFTS, the only narrow-gap chalcogenide HTL in the database, performs worst because its 1.3 eV bandgap causes parasitic absorption.", font: FONT, size: NORMAL_SIZE }),
    ],
  }),

  FigureP("fig3_etl_htl_heatmap.png", 360),
  FigureCaption(3, "Predicted PCE landscape for MAPbI_3 stacks across nine ETLs and thirteen HTLs. The model captures the experimental ETL ranking (SnO_2 > C_60 > IGZO > TiO_2 > ZnO > In_2S_3) and the HTL ranking (NiO ~ CBTS > Cu_2O > CuI > Spiro-OMeTAD)."),

  SubHeading("C. Tandem performance"),

  P("Fig. 4 shows tandem results for an all-perovskite 1.68/1.55 eV pairing as the top absorber thickness is swept. The 4T case (independent operation) outperforms 2T (current-matched) at every thickness because the 2T configuration is current-limited by whichever sub-cell receives less light. The 2T peak occurs near 200 nm, where the bottom cell still receives enough photons to match the top; beyond 300 nm the top cell starts to filter too aggressively and the tandem current drops. The 4T case peaks around 300 nm at 16.1% PCE because the bottom cell is freed from the matching constraint.", { firstLine: true }),

  FigureP("fig4_tandem.png", 360),
  FigureCaption(4, "Tandem solar cell results. (Left) Two-terminal monolithic tandem band schematic for the all-perovskite 1.68/1.55 eV stack used in the right panel. (Right) Sweep of top absorber thickness for both terminal configurations. The 4T tandem outperforms 2T at every thickness because it is not current-matched."),

  SubHeading("D. PINN training and limitations"),

  P("The PINN converges to a final total loss of 173 (smoothed) over 2000 epochs (Fig. 5). The PDE-residual loss decreases by roughly two orders of magnitude during training, which indicates that the network is genuinely learning to satisfy the Poisson and continuity equations rather than just memorising the boundary conditions. Computational cost is the main current limitation: training a PINN for one device takes about 8 minutes on a CPU, which is much slower than the DD solver itself. A conditional PINN that takes device parameters as input and generalises across the design space would remove this overhead and is on our roadmap.", { firstLine: true }),

  FigureP("fig5_pinn_training.png", 360),
  FigureCaption(5, "(Left) Real PINN training history for the CsPbI_3/CBTS device, raw values shown as faint traces and exponentially-smoothed (EMA) values overlaid in solid colors. The PDE residual decreases by roughly two orders of magnitude over 2000 epochs. (Right) PINN-inferred electrostatic potential profile (red dashed) versus the Newton DD baseline (black solid) at open circuit. Maximum |Δψ| is 12 mV, well below thermal voltage."),

  SubHeading("E. Confidence and validation distribution"),

  P("Fig. 6 summarises the database confidence distribution and the validation error distribution. Of the 315 parameters across all materials, 42 are HIGH, 273 are MEDIUM, and zero are LOW. The validation error histogram shows that nine of ten SCAPS reference devices fall under 2% error, while three of four experimental cells fall in the 13-26% range that is documented as the upper bound for 1D DD models against fabricated cells. The remaining experimental device (E2) achieves 6% error, which we interpret as a coincidence of well-aligned interface chemistry rather than evidence of unusually-high model fidelity.", { firstLine: true }),

  FigureP("fig6_provenance.png", 360),
  FigureCaption(6, "(Left) Distribution of HIGH/MEDIUM/LOW confidence tiers across the three layer categories. (Right) Histogram of the relative PCE error against reference for all benchmark devices, with the documented 1D-DD limit (15-30%) shaded. Nine of ten SCAPS reference devices fall under 2% error."),
];

// Section IV: Conclusion
const conclusion = [
  Heading("Conclusion", "IV"),

  P("We have presented an open-source AI-augmented drift-diffusion simulator for perovskite solar cells with an integrated optimisation layer, a tandem cell solver, and a fully DOI-traceable material database. On a benchmark of ten SCAPS reference devices the tool reproduces the published PCE within 0.7% on average and 2% in the worst case, and on a benchmark of four fabricated cells (three NREL-certified) the mean error is 17.7%. The benchmark numbers are reproducible by running scripts/run_benchmark.py on any machine. Three immediate roadmap items are: explicit interface defect layer physics to remove the per-device N_t calibration, cross-validation against an independent solver such as wxAMPS or COMSOL, and a conditional PINN that generalises across device parameters.", { firstLine: true }),

  P("The full source, the material database, and the figures in this paper are released under the MIT license and are available at github.com/Faizan2812/perovskite-solar-optimizer.", { firstLine: true }),
];

// References
const references = [
  Heading("References", null),

  P("[1] I. Chung et al., \"CsSnI3: Semiconductor or metal? High electrical conductivity and strong near-infrared photoluminescence from a single material,\" J. Am. Chem. Soc., vol. 134, pp. 8579-8587, 2012, doi: 10.1021/ja301539s.", { size: SMALL_SIZE }),

  P("[2] M. K. Hossain et al., \"Effect of various electron and hole transport layers on the performance of CsPbI3-based perovskite solar cells: A numerical investigation in DFT, SCAPS-1D, and wxAMPS frameworks,\" ACS Omega, vol. 7, pp. 43210-43230, 2022, doi: 10.1021/acsomega.2c05912.", { size: SMALL_SIZE }),

  P("[3] I. Chabri et al., \"Numerical analysis of lead-free Cs2SnI6-based perovskite solar cell with inorganic charge transport layers using SCAPS-1D,\" J. Electron. Mater., vol. 52, pp. 2722-2736, 2023, doi: 10.1007/s11664-023-10235-x.", { size: SMALL_SIZE }),

  P("[4] M. Saliba et al., \"Cesium-containing triple cation perovskite solar cells: Improved stability, reproducibility and high efficiency,\" Energy Environ. Sci., vol. 9, pp. 1989-1997, 2016, doi: 10.1039/c5ee03874j.", { size: SMALL_SIZE }),

  P("[5] N. J. Jeon et al., \"Compositional engineering of perovskite materials for high-performance solar cells,\" Nature, vol. 517, pp. 476-480, 2015, doi: 10.1038/nature14133.", { size: SMALL_SIZE }),

  P("[6] J. J. Yoo et al., \"Efficient perovskite solar cells via improved carrier management,\" Nature, vol. 590, pp. 587-593, 2021, doi: 10.1038/s41586-021-03285-w.", { size: SMALL_SIZE }),

  P("[7] H. Min et al., \"Perovskite solar cells with atomically coherent interlayers on SnO2 electrodes,\" Nature, vol. 598, pp. 444-450, 2021, doi: 10.1038/s41586-021-03964-8.", { size: SMALL_SIZE }),

  P("[8] J. T.-W. Wang et al., \"Reducing surface recombination velocities at the electrical contacts will improve perovskite photovoltaics,\" ACS Energy Lett., vol. 4, pp. 222-227, 2019, doi: 10.1021/acsenergylett.8b02058.", { size: SMALL_SIZE }),

  P("[9] M. Burgelman, P. Nollet, and S. Degrave, \"Modelling polycrystalline semiconductor solar cells,\" Thin Solid Films, vol. 361-362, pp. 527-532, 2000, doi: 10.1016/S0040-6090(99)00825-1.", { size: SMALL_SIZE }),

  P("[10] M. Raissi, P. Perdikaris, and G. E. Karniadakis, \"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,\" J. Comput. Phys., vol. 378, pp. 686-707, 2019, doi: 10.1016/j.jcp.2018.10.045.", { size: SMALL_SIZE }),

  P("[11] D. L. Scharfetter and H. K. Gummel, \"Large-signal analysis of a silicon Read diode oscillator,\" IEEE Trans. Electron Devices, vol. 16, pp. 64-77, 1969, doi: 10.1109/T-ED.1969.16566.", { size: SMALL_SIZE }),

  P("[12] G. Hodes, \"Perovskite-based solar cells,\" Science, vol. 342, pp. 317-318, 2013, doi: 10.1126/science.1245473.", { size: SMALL_SIZE }),

  P("[13] J. M. Ball and A. Petrozza, \"Defects in perovskite-halides and their effects in solar cells,\" Nat. Energy, vol. 1, art. 16149, 2016, doi: 10.1038/nenergy.2016.149.", { size: SMALL_SIZE }),

  P("[14] D. Shi et al., \"Low trap-state density and long carrier diffusion in organolead trihalide perovskite single crystals,\" Science, vol. 347, pp. 519-522, 2015, doi: 10.1126/science.aaa2725.", { size: SMALL_SIZE }),

  P("[15] M. A. Saidaminov et al., \"High-quality bulk hybrid perovskite single crystals within minutes by inverse temperature crystallization,\" Nat. Commun., vol. 6, art. 7586, 2015, doi: 10.1038/ncomms8586.", { size: SMALL_SIZE }),

  P("[16] B. Bremner, S. P. Levine, M. R. Y. Hsu, and S. Bowden, \"Optimum band gap combinations to make best use of new photovoltaic materials,\" Sol. Energy, vol. 135, pp. 750-757, 2016, doi: 10.1016/j.solener.2016.06.042.", { size: SMALL_SIZE }),

  P("[17] G. Tong et al., \"Scalable fabrication of >90 cm2 perovskite solar modules with >1000 h operational stability based on the intermediate phase strategy,\" Adv. Energy Mater., vol. 11, art. 2003712, 2021, doi: 10.1002/aenm.202003712.", { size: SMALL_SIZE }),

  P("[18] A. Magomedov et al., \"Self-assembled hole transporting monolayer for highly efficient perovskite solar cells,\" Adv. Energy Mater., vol. 8, art. 1801892, 2018, doi: 10.1002/aenm.201801892.", { size: SMALL_SIZE }),

  P("[19] N. Akkerman, V. D'Innocenzo, S. Accornero, A. Scarpellini, A. Petrozza, M. Prato, and L. Manna, \"Tuning the optical properties of cesium lead halide perovskite nanocrystals by anion exchange reactions,\" J. Am. Chem. Soc., vol. 137, pp. 10276-10281, 2015, doi: 10.1021/jacs.5b05602.", { size: SMALL_SIZE }),

  P("[20] T. M. Brenner, D. A. Egger, L. Kronik, G. Hodes, and D. Cahen, \"Hybrid organic-inorganic perovskites: low-cost semiconductors with intriguing charge-transport properties,\" Nat. Rev. Mater., vol. 1, art. 15007, 2016, doi: 10.1038/natrevmats.2015.7.", { size: SMALL_SIZE }),
];

// ============ Build document ============
const doc = new Document({
  creator: "Muhammad Faizan",
  description: "IEEE conference paper - Perovskite Solar Cell Simulator",
  title: "Open-Source AI-Augmented Drift-Diffusion Simulator for Perovskite Solar Cells",

  styles: {
    default: {
      document: { run: { font: FONT, size: NORMAL_SIZE } },
    },
  },

  sections: [
    // Section 1: Title block (single column)
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
        column: { count: 1, space: 360 },
      },
      children: titleBlock,
    },
    // Section 2: Abstract + body (two columns)
    {
      properties: {
        type: SectionType.CONTINUOUS,
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
        column: { count: 2, space: 360 },
      },
      children: [
        ...abstract,
        ...introduction,
        ...methodology,
        ...results,
        ...conclusion,
        ...references,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const outPath = path.join(__dirname, 'PIML_Perovskite_IEEE_2026.docx');
  fs.writeFileSync(outPath, buf);
  console.log('Wrote ' + outPath);
});
