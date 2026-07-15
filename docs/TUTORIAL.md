# Professional Tutorial & Verification Guidelines
## AI-Driven Open-Source Tool for Design & Optimization of Solar Cells

This tutorial teaches the tool feature by feature and, for every physical
result it produces, shows a **mathematically solved example** you can check
with a pen, a calculator, and the published literature. Every number quoted
here was computed against the shipped code, and the whole tutorial is
executable: `python scripts/run_tutorial_examples.py` re-derives and asserts
each worked example (Examples 1–6) automatically.

**Notation and constants used throughout** (CODATA 2018): elementary charge
$q = 1.602177\times10^{-19}$ C, Boltzmann constant
$k_B = 1.380649\times10^{-23}$ J/K, thermal voltage at $T = 300$ K:

$$V_T = \frac{k_B T}{q} = 0.025852\ \text{V}$$

Illumination is ASTM G173-03 AM1.5G, 100 mW/cm². Current densities in
mA/cm², thicknesses in nm or µm as stated.

---

# Part I — The tool, mapped completely

## 1. Validation philosophy (read this first)

Every claim in this tool rests on a four-layer verification ladder, and this
tutorial walks each rung with solved mathematics:

1. **Analytic hand-verification** — closed-form physics (diode equation,
   detailed balance, Fresnel optics, depletion approximation) must reproduce
   the engine outputs. Part II of this tutorial.
2. **Independent internal code paths** — two implementations of the same
   physics must agree (TMM optics vs calibrated optics; Richter-2012 vs
   Niewelt-2022 recombination; PINN vs finite-difference drift-diffusion).
3. **Certified published devices** — 14/14 experimental benchmarks with DOIs,
   mean PCE error 0.73% (`VALIDATION_MULTI_TECH.md`).
4. **External tools** — SCAPS-1D numerical cross-check (10 published devices,
   mean 0.7% error) and Lumerical-class TMM optical benchmarks, 9/9 checks
   (`docs/CROSS_TOOL_VALIDATION.md`).

Layers 3–4 already run automatically. This tutorial adds layer 1 explicitly,
because a researcher who can reproduce a result by hand owns it.

## 2. Installation and first run

```bash
git clone <your-repo-url>
cd perovskite-solar-optimizer
pip install -r requirements.txt
streamlit run app.py                          # launches the web UI
python scripts/run_multi_tech_validation.py   # 14/14 experimental benchmarks
python scripts/run_cross_tool_validation.py   # 9/9 cross-tool checks
python scripts/run_tutorial_examples.py       # this tutorial's math, asserted
pytest tests/ -q                              # full regression suite
```

The Streamlit sidebar lists the main app plus eleven feature pages. The
repository layout: `physics/` (all device engines), `ai/` (PINNs, ML
surrogates, optimizers), `utils/` (interop, datasets, benchmarks), `data/`
(material + reference databases with per-parameter provenance), `scripts/`
(validation suites), `artifacts/` (versioned results every claim regenerates
from), `tests/` (regression suite), `docs/` (validation and interop reports).

## 3. Every feature, explained

**Main app (`app.py`)** — the perovskite workbench: stack builder
(HTL/absorber/ETL selection from the provenance-tracked database), the fast
1-D simulator (J-V, EQE, band diagram, loss analysis), the full
finite-difference drift-diffusion solver for publication-grade runs,
temperature and defect-density studies, and the perovskite/perovskite
tandem tab.

**Page 1 · Advanced Physics** — the numerical drift-diffusion core exposed:
Scharfetter-Gummel discretization, mesh and convergence controls,
recombination channel breakdown (radiative / SRH / Auger / interface), and
energy-band visualization. Use this page when a reviewer asks "what exactly
does your solver solve?" — the answer is the Poisson + electron/hole
continuity system of Eq. (18)–(20) below.

**Page 2 · Integrity & Validation** — the tool grades itself: the published
SCAPS-1D replication suite, the experimental-device benchmark table, and the
reference audit (which citations are machine-verified). Nothing on this page
is hand-typed; it renders the JSON artifacts the scripts produce.

**Page 3 · Conditional PINN** — a physics-informed neural network
$\hat{u}_\theta(x; d, N_t)$ trained across a *family* of devices
(thickness 300–700 nm, defect density $10^{13}$–$10^{16}$ cm⁻³) rather than
one device, so it interpolates new designs in milliseconds. Verified against
the finite-difference solver (Example 6).

**Page 4 · CdTe Technology** — the second technology family
(glass/SnO₂/CdS/CdTe superstrate) on the same drift-diffusion core,
calibrated to the community baseline (Gloeckler 2003) and validated
out-of-sample against Ahmed 2023.

**Page 5 · Inverse Design (PINN)** — gradient-based inversion: specify a
target (e.g. Voc ≥ 1.15 V at PCE ≥ 20%) and the differentiable surrogate
searches the design space backwards, returning candidate stacks with their
forward-verified metrics.

**Page 6 · Optimizer Benchmark** — GA, differential evolution, NSGA-II and
Bayesian optimization raced on identical budgets against known optima, so
optimizer claims are measured, not asserted.

**Page 7 · Datasets & Interop** — Perovskite Database Project ingestion for
ML training, SCAPS-1D `.def` import **and** export, generic device-spec JSON,
J-V/EQE CSV. The round-trip is regression-tested (Part III, §14).

**Page 8 · Silicon Technology** — five architecture presets (Al-BSF,
PERC, TOPCon, SHJ, SHJ-IBC) validated against certified record devices;
design explorer over wafer thickness, doping, lifetime, surface saturation
current, series resistance; selectable Richter-2012/Niewelt-2022 intrinsic
recombination; DE optimization with random-forest sensitivity ranking.

**Page 9 · Organic Technology** — six certified-device benchmarks
(15.7% PM6:Y6 through 20.4% PM6:L8-BO-C4 plus P3HT:PCBM), thickness roll-off
explorer, the 🌊 Wave-optics (TMM) tab with per-layer parasitic absorption
and the interference Jsc(L) curve, and DE optimization.

**Page 10 · Tandem Designer** — perovskite/Si, perovskite/organic and
perovskite/perovskite tandems, 2T (true series J-V addition) and 4T modes,
spectral-transmission plots, current-matching thickness scans, and the
record-lineage validation (29.15% → 34.58%).

**Page 11 · References Library** — every citation in the tool (37 in the v4
registry plus the material-database references), searchable, with DOI,
confidence tier and verification status, exportable to CSV.

---

# Part II — Mathematically solved verification examples

Each example states the physics, solves it numerically by hand, and compares
against the tool. All tool outputs below are exact values from the shipped
code (regenerate with `scripts/run_tutorial_examples.py`).

## Example 1 — Silicon SHJ-IBC: the full efficiency, by hand

**Device**: the "SHJ-IBC (Kaneka 26.7%)" preset — n-type wafer,
$W = 165$ µm, $N_D = 3\times10^{15}$ cm⁻³, bulk SRH lifetime
$\tau = 15$ ms, total surface saturation current
$J_{0s} = 2.6$ fA/cm², $R_s = 0.32\ \Omega$cm², Lambertian path factor
$Z = 50$, front reflectance 1.2%. Certified target: 26.7%
(Yoshikawa 2017, DOI 10.1016/j.solmat.2017.06.024).

**Step 1 — Photocurrent bound.** The AM1.5G photon flux integrated over the
silicon absorption window (300–1200 nm) is
$2.90\times10^{17}$ photons/cm²/s, equivalent to a ceiling of

$$J_{sc}^{max} = q\,\Phi = 46.41\ \text{mA/cm}^2.$$

The engine's optical stack (Lambertian absorptance
$A = (1-R_f)\,\alpha/(\alpha + 1/ZW)$, Tiedje-Yablonovitch 1984, with Green-2008
$\alpha(\lambda)$) delivers $J_{sc} = 43.00$ mA/cm² — an optical utilization
of $43.00/46.41 = 92.7\%$, exactly what near-unity-EQE record silicon
achieves (Yoshikawa's device: 42.65).

**Step 2 — Open-circuit voltage from the recombination balance.** At open
circuit, generation equals recombination. The engine reports the average
generation rate $G = J_{sc}/(qW) = 1.6264\times10^{19}$ cm⁻³s⁻¹ and solves
for the excess density $\Delta n$. Verify the balance at the engine's
solution $\Delta n = 1.605\times10^{16}$ cm⁻³ channel by channel:

*Intrinsic (Richter 2012, Eq. 18)*: with $n_0 = 3\times10^{15}$,
$p_0 = n_i^2/N_D = 3.10\times10^{4}$ cm⁻³
($n_i = 9.65\times10^9$, Altermatt 2003), $n = n_0 + \Delta n$,
$p \approx \Delta n$:

$$R_{intr} = np\left(2.5\times10^{-31} g_{eeh} n_0 + 8.5\times10^{-32} g_{ehh} p_0 + 3.0\times10^{-29}\Delta n^{0.92}\right) + B(np - n_i^2) = 1.196\times10^{19}$$

*SRH*: $R_{SRH} = \Delta n/\tau = 1.605\times10^{16}/0.015 =
1.070\times10^{18}$.

*Surfaces*: $R_{surf} = \dfrac{J_{0s}}{qW}\left(\dfrac{np}{n_i^2}-1\right) =
3.230\times10^{18}$.

Sum: $1.196\times10^{19} + 0.107\times10^{19} + 0.323\times10^{19} =
1.626\times10^{19} = G$ ✓ — the balance closes to four significant figures.
The voltage follows from the quasi-Fermi splitting:

$$V_{oc} = V_T \ln\frac{(n_0+\Delta n)(p_0+\Delta n)}{n_i^2}
= 0.025852 \times \ln\!\left(\frac{1.905\times10^{16}\times 1.605\times10^{16}}{(9.65\times10^{9})^2}\right) = 0.025852 \times 28.82 = \mathbf{0.7451\ V}.$$

Tool output: $V_{oc} = 0.7451$ V. Exact agreement, and 0.9% from the
certified 0.738 V.

**Step 3 — Fill factor (Green 1981).** With normalized voltage
$v_{oc} = V_{oc}/(m V_T) = 28.82$ (ideality $m = 1.0$ here):

$$FF_0 = \frac{v_{oc} - \ln(v_{oc} + 0.72)}{v_{oc}+1} = \frac{28.82 - 3.386}{29.82} = 0.8529.$$

Series-resistance correction with $r_s = R_s J_{sc}/V_{oc} =
0.32\times43.00/745.1 = 0.0185$:

$$FF \approx FF_0(1 - 1.1 r_s) = 0.8529 \times 0.9796 = \mathbf{0.8356}.$$

Tool (full Newton J-V sweep): $FF = 0.8357$. Agreement to $10^{-4}$.

**Step 4 — Efficiency.**

$$\eta = \frac{V_{oc} J_{sc} FF}{P_{in}} = \frac{0.7451\times43.00\times0.8357}{100} = \mathbf{26.77\%}$$

against the certified 26.7% (error 0.26%). Every factor of this number has
now been derived by hand.

## Example 2 — Organic PM6:Y6: energy-loss Voc, spectral Jsc, diode FF

**Device**: PM6:Y6 preset, 100 nm active layer. Certified target: 15.7%,
$V_{oc} = 0.83$ V, $J_{sc} = 25.3$, $FF = 74.8\%$ (Yuan 2019,
DOI 10.1016/j.joule.2019.01.004).

**Step 1 — Voc from the energy-loss picture** (Scharber 2006 lineage). Y6's
optical gap is $E_g = 1.33$ eV and state-of-the-art NFA blends lose
$E_{loss} = 0.50$ eV to radiative + non-radiative channels:

$$V_{oc} = \frac{E_g - E_{loss}}{q} = 1.33 - 0.50 = \mathbf{0.830\ V}.$$

Tool: 0.829 V (the 1 mV difference is the J-V zero-crossing interpolation).

**Step 2 — Jsc as a windowed spectral integral.** The absorption window runs
from 350 nm to the optical edge $\lambda_{edge} = 1240/1.33 = 932$ nm. The
AM1.5G photon current in that window is

$$J_{window} = q\int_{350}^{932}\Phi(\lambda)\,d\lambda = 34.90\ \text{mA/cm}^2.$$

With the measured plateau EQE of 0.769 (calibrated to the certified device):
$34.90 \times 0.769 = 26.84$ mA/cm². The near-edge roll-off (measured NFA EQE
decays over the last ~150 nm; modeled as a raised ramp) removes a further
5.8%, giving $\mathbf{25.29}$ — the tool's value, against 25.3 certified.

**Step 3 — Diode parameters and FF.** The saturation current consistent with
this $(V_{oc}, J_{sc})$ pair at ideality $m = 1.52$:

$$J_0 = \frac{J_{sc}}{e^{V_{oc}/mV_T} - 1} = \frac{25.29}{e^{0.829/0.0393}-1} = \mathbf{1.74\times10^{-8}\ \text{mA/cm}^2},$$

matching the engine's $1.70\times10^{-8}$ (2%, from the Voc interpolation).
Green's expression with $v_{oc} = 21.1$ gives $FF_0 = 0.8152$; the resistive
losses ($r_s = 0.067$, plus the 1.4 kΩcm² shunt) bring the full sweep to
$FF = 0.7424$ — vs 0.748 certified. PCE: $0.829\times25.29\times0.7424 =
\mathbf{15.57\%}$ (certified 15.7%, error 0.84%).

**Cross-path check**: switching to rigorous wave optics,
`simulate_organic(..., optics="tmm")` yields 14.88% — a 4.4% deviation
between two *independent* optical models, well inside the ~16% average
TMM-vs-experiment band published by Rosa 2021.

## Example 3 — Perovskite MAPbI₃: absorption, Jsc bound, and the detailed-balance ceiling

**Device**: glass/SnO₂(50)/MAPbI₃(500 nm, $N_t = 10^{14}$)/Spiro-OMeTAD(150).
Tool output: $V_{oc} = 1.038$ V, $J_{sc} = 21.47$, $FF = 0.857$,
PCE = 19.10%.

**Step 1 — Tauc absorption by hand.** MAPbI₃: $E_g = 1.55$ eV,
$\alpha_0 = 1.5\times10^5$ cm⁻¹eV$^{-1/2}$. At $\lambda = 600$ nm
($E = 2.066$ eV):

$$\alpha = \alpha_0\frac{\sqrt{E - E_g}}{E} = 1.5\times10^5 \times \frac{\sqrt{0.516}}{2.066} = 5.22\times10^4\ \text{cm}^{-1},$$

so a 500 nm film absorbs $A = 1 - e^{-\alpha d} = 1 - e^{-2.61} = 0.926$ of
600 nm light in a single pass — why 500 nm suffices for perovskites while
silicon needs 160 µm.

**Step 2 — Photocurrent bound.** Above-gap AM1.5G flux
($\lambda \le hc/E_g = 800$ nm) gives $J_{sc}^{max} = 26.91$ mA/cm². The tool's
21.47 corresponds to 79.8% optical+collection utilization at $N_t = 10^{14}$ —
record MAPbI₃ cells reach ~24; the gap is the deliberately non-ideal defect
density, which you can verify by lowering $N_t$ in the UI.

**Step 3 — The Shockley-Queisser Voc ceiling (sanity bound).** In the
Boltzmann approximation, the radiative saturation current for $E_g = 1.55$ eV:

$$J_{0,rad} \approx q\,\frac{2\pi k_B T\,E_g^2}{h^3 c^2}\,e^{-E_g/k_B T} = 8.99\times10^{-21}\ \text{mA/cm}^2,$$

$$V_{oc}^{rad} = V_T\ln\!\left(\frac{J_{sc}^{max}}{J_{0,rad}}+1\right) = 0.025852\times\ln(3.03\times10^{21}) = \mathbf{1.279\ V}.$$

The tool's 1.038 V sits 241 mV below the radiative limit — precisely the
non-radiative penalty of $N_t = 10^{14}$ cm⁻³ SRH centers. Any simulator
returning $V_{oc} > 1.28$ V for MAPbI₃ would be violating thermodynamics;
this bound is your fastest smoke test of *any* PV tool.

## Example 4 — Tandem (2020-grade): spectral splitting and series addition

**Device**: 1.68 eV perovskite (650 nm) on SHJ-IBC silicon, $N_t =
2\times10^{14}$, interconnect $R_s = 4.5\ \Omega$cm². Certified target:
29.15%, $V_{oc} = 1.92$ V (Al-Ashouri 2020, DOI 10.1126/science.abd4016).

**Step 1 — Voltage additivity.** At open circuit no current flows, so the
interconnect drops nothing and the 2T voltage must be the subcell sum. Tool
subcells: top 1.201 V, filtered-spectrum bottom 0.727 V. Hand sum: 1.928 V;
tool tandem $V_{oc} = 1.922$ V — additivity verified to 0.3% (the residual
is J-V grid interpolation). Note the bottom cell's Voc fell from 0.745 V
(full sun, Example 1) to 0.727 V: the logarithmic intensity penalty
$V_T\ln(19.44/43.00) = -20.5$ mV, hand-checked.

**Step 2 — Current limiting.** Top $J_{sc} = 17.90$, bottom (under the
top-filtered spectrum) 19.44 mA/cm². A series stack conducts the smaller:
tandem $J_{sc} = 17.90$ ✓. The mismatch is why the summed-curve fill factor
(0.847) exceeds a matched stack's: the bottom cell operates below its own
MPP current, contributing nearly flat voltage.

**Step 3 — Efficiency.** $1.922 \times 17.90 \times 0.8474 / 100 =
\mathbf{29.16\%}$ vs 29.15% certified. The engine's disclosed trade-off
(Jsc ~7% low, FF ~6% high, compensating) is printed in
`VALIDATION_MULTI_TECH.md` — hand-verify it yourself against the paper's
19.26 mA/cm² and 79.5%.

## Example 5 — Wave optics (TMM): Fresnel by hand, interference by design rule

**Check 1 — Fresnel limit.** A bare glass/air interface at normal incidence:

$$R = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2 = \left(\frac{0.52}{2.52}\right)^2 = 0.042580.$$

`tmm_single` returns 0.04257999… and $R + T = 1$ to machine precision —
the solver reduces exactly to the analytic solution where one exists.

**Check 2 — Quarter-wave design rule.** Constructive interference at the
reflective back contact places the first absorption maximum near
$d \approx \lambda_{peak}/4n$. For P3HT:PCBM ($n = 2.08$ at its 550 nm
absorption peak): $d \approx 550/(4\times2.08) = 66$ nm. The full broadband
TMM scan puts the maximum at 80 nm — the shift is the AM1.5G spectral
weighting plus the non-ideal metal phase, and the published TMM literature
(Sievers 2006; Monestier 2007) reports the same 70–90 nm window. The second
maximum (215 nm, 10.1 mA/cm²) and the 130 nm minimum likewise sit inside
every published window — this oscillation is the classic Lumerical-class
signature, reproduced quantitatively.

**Check 3 — Energy conservation.** For every wavelength the solver satisfies
$R + T + \sum_j A_j = 1$; the shipped stacks close the budget to
$<10^{-4}$, the discretization floor of the field integral.

## Example 6 — PINN verification: how a neural network earns trust

The Conditional PINN (page 3) approximates the drift-diffusion solution
$u = (\psi, \log n, \log p)(x; d, N_t)$ across a device family by minimizing
a composite loss

$$\mathcal{L} = \underbrace{\lVert \hat u_\theta - u_{FDM}\rVert^2}_{\text{data}} + \lambda \underbrace{\left\lVert \frac{d}{dx}\!\left(\varepsilon\frac{d\hat\psi}{dx}\right) + q\,(p - n + N_D - N_A)\right\rVert^2}_{\text{Poisson residual}} + \dots$$

Verification is *never* "the loss went down"; it is a three-part protocol
with numbers you can reproduce:

**(a) Against the validated numerical solver.** The shipped artifact
(`artifacts/conditional_pinn_metrics.json`) reports relative-L2 errors on
*held-out* devices the network never trained on: 7.3% ($\psi$) / 7.5%
($\log n$) at $d=500$ nm, $N_t=10^{14}$; 6.5%/6.5% at 400 nm,
$3\times10^{15}$; 7.6%/7.4% at 600 nm, $10^{15}$ — interpolation across the
family at ≲8% field error, with the final data loss $8.2\times10^{-3}$.

**(b) Against closed-form physics.** Two analytic anchors bound any 1-D
solution. First, the depletion width: for a junction with
$\varepsilon_r = 25$, $V_{bi} = 1$ V, $N = 10^{15}$ cm⁻³,

$$W_D = \sqrt{\frac{2\varepsilon_0\varepsilon_r V_{bi}}{qN}} = 1662\ \text{nm} \gg 500\ \text{nm},$$

so a 500 nm perovskite absorber is *fully depleted* and the equilibrium
field must be nearly uniform, $E \approx V_{bi}/d = 20$ kV/cm — check the
PINN's $d\psi/dx$ against this by eye on page 3. Second, thermal
equilibrium demands the Boltzmann relation
$n\,p = n_i^2\,e^{(E_{Fn}-E_{Fp})/k_BT}$ with zero quasi-Fermi splitting at
$V = 0$ in the dark: any pointwise violation is a pointwise error meter
needing no reference solution at all.

**(c) Against the forward observable.** Fields are means; J-V is the end.
The inverse-design page always re-simulates PINN-proposed devices through
the finite-difference solver before reporting them — a PINN suggestion is
never itself the evidence.

This protocol — reference solution, analytic anchors, forward re-simulation
— is the template for verifying *any* surrogate you train with this tool.

---

# Part III — Working like a professional

## 7. The universal verification checklist

For any result $R$ this tool (or any tool) gives you, run the ladder:
(1) bound it analytically — is $J_{sc}$ below the above-gap photon current?
Is $V_{oc}$ below the radiative limit of Example 3? Is FF below Green's
$FF_0(v_{oc})$? (2) reproduce it through an independent path — TMM vs
calibrated optics, Niewelt vs Richter, PINN vs FDM; (3) place it against a
certified device from the References page; (4) export it
(SCAPS `.def`, device JSON, CSV) and re-run it in the incumbent tool. A
result that survives all four is publishable; the scripts in `scripts/`
automate rungs 2–4.

## 8. AI features, correctly used

**Optimizers** (pages 6, 8, 9; `ai/multi_tech_optimizer.py`): differential
evolution mutates a population of parameter vectors,
$x' = x_a + F(x_b - x_c)$, and needs no gradients — right for the
non-smooth, multi-modal PV design landscape. Always check the best-so-far
convergence plot: a curve still rising at budget exhaustion means increase
`maxiter`. The random-forest sensitivity ranking fitted on the evaluated
population tells you *which* knob mattered — report that, not just the
optimum.

**Surrogates & uncertainty** (`ai/ml_models.py`, `ai/uncertainty.py`):
ensemble disagreement flags extrapolation; treat any prediction whose
uncertainty band exceeds the effect you are claiming as "measure first".

**Inverse design** (page 5): treat outputs as *candidates*; the page's
forward re-simulation column is the verdict.

**Physical plausibility rule**: optimizers will happily drive parameters to
unphysical corners ($E_{loss} < 0.5$ eV, $\tau \to \infty$). The engines
clamp hard thermodynamic bounds (silicon's 29.4% test is in the suite), but
material-realism judgment stays with you — the tutorial's Example 3 bound
is your fastest filter.

## 9. Interoperability walkthrough

Design a device → download the SCAPS `.def` (layer thicknesses, $E_g$,
$\chi$, $\varepsilon$, $N_c$, $N_v$, mobilities, doping, dominant defect all
transfer; skipped SCAPS-extras are listed in the file header) → open in
SCAPS-1D and run → compare against the exported J-V CSV. The reverse path
(`Datasets & Interop` page) imports any published `.def`. The device-spec
JSON serves OghmaNano and scripted pipelines. Round-trip exactness is
enforced by `tests/test_multi_technology.py::test_interop_roundtrip`.

## 10. Reproducing and extending the validation

`run_multi_tech_validation.py` and `run_cross_tool_validation.py` regenerate
every headline number into `artifacts/*.json`. To add your own certified
benchmark: append a `references` entry (citation + DOI + confidence) and a
`benchmarks` entry (targets + tolerance) to
`data/multi_technology_database.json` — the registry-integrity test will
refuse benchmarks citing unregistered sources, and both the validation
script and the technology pages pick the new row up automatically.

## 11. Known limits (what this tool will honestly refuse to claim)

1-D physics only (no busbar geometry, no 3-D textures — export to
Quokka/FDTD for those); organic transport is semi-empirical (exciton/CT
drift-diffusion → OghmaNano); TMM assumes planar coherent stacks; organic
n,k tables carry ±5–10% batch spread; tandem grade presets are calibrated
within documented ranges, not ab-initio. Each limit, its literature basis,
and its workaround are catalogued in `MAJOR_CONCERNS.md`.

## 12. Citing

Cite the tool via `CITATION.cff`, and cite the *original physics* whose DOIs
travel with every number — the References page exports the full list. A
suggested methods sentence: "Device simulations were performed with the
AI-Driven Open-Source Solar Cell Design Tool final-release (validated against 14
certified record devices, mean PCE error 0.73%, and cross-checked against
SCAPS-1D and transfer-matrix optics, 9/9 checks); model equations and
parameter provenance follow the references therein."
