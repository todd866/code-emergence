// =============================================================================
// GENERATE PUBLICATION FIGURES FOR JTB PAPER
// Run with: node generate_figures.js
// Outputs: CSV data files + LaTeX-ready tables
// =============================================================================

const fs = require('fs');

console.log("=== GENERATING JTB PAPER FIGURES ===\n");

const N = 64;
const nTrials = 15;  // More trials for cleaner data
const burnIn = 800;
const measureSteps = 800;
const dt = 0.1;

// =============================================================================
// KURAMOTO LATTICE
// =============================================================================

function createLattice() {
  const theta = new Float32Array(N);
  const omega = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    theta[i] = Math.random() * 2 * Math.PI - Math.PI;
    omega[i] = 0.5 + (Math.random() - 0.5) * 0.3;
  }
  return { theta, omega };
}

function stepLattice(lattice, K, noise, dt) {
  const { theta, omega } = lattice;
  const sqrtDt = Math.sqrt(dt);  // Euler-Maruyama: noise scales with sqrt(dt)

  for (let i = 0; i < N; i++) {
    const left = (i - 1 + N) % N;
    const right = (i + 1) % N;
    let coupling = Math.sin(theta[left] - theta[i]) + Math.sin(theta[right] - theta[i]);
    const u1 = Math.random(), u2 = Math.random();
    const eta = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    // Deterministic terms scale with dt; stochastic term scales with sqrt(dt)
    theta[i] += dt * (omega[i] + K * coupling) + sqrtDt * noise * eta;
    theta[i] = ((theta[i] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }
}

// =============================================================================
// FOURIER CODE
// =============================================================================

function fourierEncode(theta, maxK) {
  const modes = [];
  for (let k = 0; k <= maxK; k++) {
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) {
      const phase = theta[i] - 2 * Math.PI * k * i / N;
      re += Math.cos(phase);
      im += Math.sin(phase);
    }
    modes.push({ re: re / N, im: im / N });
  }
  return modes;
}

function fourierDecode(modes, k) {
  const recon = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let zRe = 0, zIm = 0;
    for (let m = 0; m <= Math.min(k, modes.length - 1); m++) {
      const phase = 2 * Math.PI * m * i / N;
      const c = Math.cos(phase), s = Math.sin(phase);
      zRe += modes[m].re * c - modes[m].im * s;
      zIm += modes[m].re * s + modes[m].im * c;
      if (m > 0) {
        zRe += modes[m].re * c + modes[m].im * s;
        zIm += -modes[m].re * s + modes[m].im * c;
      }
    }
    recon[i] = Math.atan2(zIm, zRe);
  }
  return recon;
}

// =============================================================================
// METRICS
// =============================================================================

function spectralComplexity(theta) {
  const modes = fourierEncode(theta, N / 2);
  const amps = [];
  let sum = 0;
  for (let k = 1; k < modes.length; k++) {
    const amp = Math.sqrt(modes[k].re ** 2 + modes[k].im ** 2);
    amps.push(amp);
    sum += amp;
  }
  if (sum < 1e-10) return 1;
  let entropy = 0;
  for (const a of amps) {
    const p = a / sum;
    if (p > 1e-10) entropy -= p * Math.log(p);
  }
  return Math.exp(entropy);
}

function phaseMismatch(thetaA, thetaB) {
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const diff = thetaA[i] - thetaB[i];
    sum += Math.abs(Math.sin(diff / 2));
  }
  return sum / N;
}

// =============================================================================
// COUPLED STEP
// =============================================================================

function stepCoupled(A, B, k, K, lambda, noise, dt) {
  stepLattice(A, K, noise, dt);

  const modesA = fourierEncode(A.theta, k);
  const targetPhase = fourierDecode(modesA, k);

  const { theta, omega } = B;
  const sqrtDt = Math.sqrt(dt);  // Euler-Maruyama: noise scales with sqrt(dt)

  for (let i = 0; i < N; i++) {
    const left = (i - 1 + N) % N;
    const right = (i + 1) % N;
    let coupling = Math.sin(theta[left] - theta[i]) + Math.sin(theta[right] - theta[i]);
    const codeConstraint = lambda * Math.sin(targetPhase[i] - theta[i]);
    const u1 = Math.random(), u2 = Math.random();
    const eta = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    // Deterministic terms scale with dt; stochastic term scales with sqrt(dt)
    theta[i] += dt * (omega[i] + K * coupling + codeConstraint) + sqrtDt * noise * 0.5 * eta;
    theta[i] = ((theta[i] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }
}

// =============================================================================
// FIGURE 3: COMPLEXITY COLLAPSE (k sweep)
// =============================================================================

console.log("=== FIGURE 3: COMPLEXITY vs k ===\n");

const K = 0.5;
const noise = 0.3;
const lambda_fixed = 1.0;

const kValues = [1, 2, 4, 8, 16, 32];
const fig3Data = [];

for (const k of kValues) {
  let totalComplexA = 0, totalComplexB = 0, totalMismatch = 0;
  let complexAVals = [], complexBVals = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const A = createLattice();
    const B = createLattice();

    for (let t = 0; t < burnIn; t++) {
      stepCoupled(A, B, k, K, lambda_fixed, noise, dt);
    }

    let sumComplexA = 0, sumComplexB = 0, sumMismatch = 0;
    for (let t = 0; t < measureSteps; t++) {
      stepCoupled(A, B, k, K, lambda_fixed, noise, dt);
      sumComplexA += spectralComplexity(A.theta);
      sumComplexB += spectralComplexity(B.theta);
      sumMismatch += phaseMismatch(A.theta, B.theta);
    }

    const trialComplexA = sumComplexA / measureSteps;
    const trialComplexB = sumComplexB / measureSteps;

    totalComplexA += trialComplexA;
    totalComplexB += trialComplexB;
    totalMismatch += sumMismatch / measureSteps;

    complexAVals.push(trialComplexA);
    complexBVals.push(trialComplexB);
  }

  const avgComplexA = totalComplexA / nTrials;
  const avgComplexB = totalComplexB / nTrials;
  const avgMismatch = totalMismatch / nTrials;

  // Standard error
  const seA = Math.sqrt(complexAVals.reduce((s, v) => s + (v - avgComplexA) ** 2, 0) / (nTrials - 1)) / Math.sqrt(nTrials);
  const seB = Math.sqrt(complexBVals.reduce((s, v) => s + (v - avgComplexB) ** 2, 0) / (nTrials - 1)) / Math.sqrt(nTrials);

  fig3Data.push({ k, complexA: avgComplexA, complexB: avgComplexB, mismatch: avgMismatch, seA, seB });

  console.log(`k=${k.toString().padStart(2)}: N_eff(A)=${avgComplexA.toFixed(2)}±${seA.toFixed(2)} N_eff(B)=${avgComplexB.toFixed(2)}±${seB.toFixed(2)} mismatch=${avgMismatch.toFixed(3)}`);
}

// =============================================================================
// FIGURE 4: COUPLING STRENGTH (lambda sweep)
// =============================================================================

console.log("\n=== FIGURE 4: COMPLEXITY vs lambda ===\n");

const lambdaValues = [0, 0.25, 0.5, 1.0, 2.0, 4.0];
const k_fixed = 8;
const fig4Data = [];

for (const lambda of lambdaValues) {
  let totalComplexB = 0, totalMismatch = 0;

  for (let trial = 0; trial < nTrials; trial++) {
    const A = createLattice();
    const B = createLattice();

    for (let t = 0; t < burnIn; t++) {
      stepCoupled(A, B, k_fixed, K, lambda, noise, dt);
    }

    let sumComplexB = 0, sumMismatch = 0;
    for (let t = 0; t < measureSteps; t++) {
      stepCoupled(A, B, k_fixed, K, lambda, noise, dt);
      sumComplexB += spectralComplexity(B.theta);
      sumMismatch += phaseMismatch(A.theta, B.theta);
    }

    totalComplexB += sumComplexB / measureSteps;
    totalMismatch += sumMismatch / measureSteps;
  }

  const avgComplexB = totalComplexB / nTrials;
  const avgMismatch = totalMismatch / nTrials;

  fig4Data.push({ lambda, complexB: avgComplexB, mismatch: avgMismatch });

  console.log(`λ=${lambda.toFixed(2).padStart(4)}: N_eff(B)=${avgComplexB.toFixed(2)} mismatch=${avgMismatch.toFixed(3)}`);
}

// =============================================================================
// CONTROL: No coupling
// =============================================================================

console.log("\n=== CONTROL: lambda=0 ===\n");

let controlComplexB = 0, controlMismatch = 0;
for (let trial = 0; trial < nTrials; trial++) {
  const A = createLattice();
  const B = createLattice();

  for (let t = 0; t < burnIn; t++) {
    stepLattice(A, K, noise, dt);
    stepLattice(B, K, noise, dt);
  }

  let sumComplexB = 0, sumMismatch = 0;
  for (let t = 0; t < measureSteps; t++) {
    stepLattice(A, K, noise, dt);
    stepLattice(B, K, noise, dt);
    sumComplexB += spectralComplexity(B.theta);
    sumMismatch += phaseMismatch(A.theta, B.theta);
  }

  controlComplexB += sumComplexB / measureSteps;
  controlMismatch += sumMismatch / measureSteps;
}

console.log(`Control (uncoupled): N_eff(B)=${(controlComplexB / nTrials).toFixed(2)} mismatch=${(controlMismatch / nTrials).toFixed(3)}`);

// =============================================================================
// CONTROL: RANDOM PROJECTION (same k, different basis)
// =============================================================================

console.log("\n=== CONTROL: RANDOM PROJECTION ===\n");
console.log("(Using random k modes instead of lowest k Fourier modes)\n");

// Generate a fixed random projection matrix for reproducibility
function createRandomProjection(k) {
  // Select k random mode indices from [1, N/2]
  const allModes = [];
  for (let m = 1; m <= N / 2; m++) allModes.push(m);
  // Fisher-Yates shuffle
  for (let i = allModes.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allModes[i], allModes[j]] = [allModes[j], allModes[i]];
  }
  return allModes.slice(0, k);
}

function fourierEncodeRandom(theta, randomModes) {
  const modes = [];
  for (const k of randomModes) {
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) {
      const phase = theta[i] - 2 * Math.PI * k * i / N;
      re += Math.cos(phase);
      im += Math.sin(phase);
    }
    modes.push({ k, re: re / N, im: im / N });
  }
  return modes;
}

function fourierDecodeRandom(modes) {
  const recon = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let zRe = 0, zIm = 0;
    for (const m of modes) {
      const phase = 2 * Math.PI * m.k * i / N;
      const c = Math.cos(phase), s = Math.sin(phase);
      // Add both positive and negative frequency components
      zRe += m.re * c - m.im * s;
      zIm += m.re * s + m.im * c;
      zRe += m.re * c + m.im * s;
      zIm += -m.re * s + m.im * c;
    }
    recon[i] = Math.atan2(zIm, zRe);
  }
  return recon;
}

function stepCoupledRandom(A, B, randomModes, K, lambda, noise, dt) {
  stepLattice(A, K, noise, dt);

  const modesA = fourierEncodeRandom(A.theta, randomModes);
  const targetPhase = fourierDecodeRandom(modesA);

  const { theta, omega } = B;
  const sqrtDt = Math.sqrt(dt);  // Euler-Maruyama: noise scales with sqrt(dt)

  for (let i = 0; i < N; i++) {
    const left = (i - 1 + N) % N;
    const right = (i + 1) % N;
    let coupling = Math.sin(theta[left] - theta[i]) + Math.sin(theta[right] - theta[i]);
    const codeConstraint = lambda * Math.sin(targetPhase[i] - theta[i]);
    const u1 = Math.random(), u2 = Math.random();
    const eta = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    // Deterministic terms scale with dt; stochastic term scales with sqrt(dt)
    theta[i] += dt * (omega[i] + K * coupling + codeConstraint) + sqrtDt * noise * 0.5 * eta;
    theta[i] = ((theta[i] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }
}

const randomProjData = [];

for (const k of kValues) {
  let totalComplexA = 0, totalComplexB = 0, totalMismatch = 0;
  let complexAVals = [], complexBVals = [];

  for (let trial = 0; trial < nTrials; trial++) {
    const A = createLattice();
    const B = createLattice();
    const randomModes = createRandomProjection(k);

    for (let t = 0; t < burnIn; t++) {
      stepCoupledRandom(A, B, randomModes, K, lambda_fixed, noise, dt);
    }

    let sumComplexA = 0, sumComplexB = 0, sumMismatch = 0;
    for (let t = 0; t < measureSteps; t++) {
      stepCoupledRandom(A, B, randomModes, K, lambda_fixed, noise, dt);
      sumComplexA += spectralComplexity(A.theta);
      sumComplexB += spectralComplexity(B.theta);
      sumMismatch += phaseMismatch(A.theta, B.theta);
    }

    const trialComplexA = sumComplexA / measureSteps;
    const trialComplexB = sumComplexB / measureSteps;

    totalComplexA += trialComplexA;
    totalComplexB += trialComplexB;
    totalMismatch += sumMismatch / measureSteps;

    complexAVals.push(trialComplexA);
    complexBVals.push(trialComplexB);
  }

  const avgComplexA = totalComplexA / nTrials;
  const avgComplexB = totalComplexB / nTrials;
  const avgMismatch = totalMismatch / nTrials;

  const seA = Math.sqrt(complexAVals.reduce((s, v) => s + (v - avgComplexA) ** 2, 0) / (nTrials - 1)) / Math.sqrt(nTrials);
  const seB = Math.sqrt(complexBVals.reduce((s, v) => s + (v - avgComplexB) ** 2, 0) / (nTrials - 1)) / Math.sqrt(nTrials);

  randomProjData.push({ k, complexA: avgComplexA, complexB: avgComplexB, mismatch: avgMismatch, seA, seB });

  console.log(`k=${k.toString().padStart(2)}: N_eff(A)=${avgComplexA.toFixed(2)}±${seA.toFixed(2)} N_eff(B)=${avgComplexB.toFixed(2)}±${seB.toFixed(2)} mismatch=${avgMismatch.toFixed(3)}`);
}

// =============================================================================
// OUTPUT CSV FILES
// =============================================================================

console.log("\n=== WRITING CSV FILES ===\n");

// Figure 3 data
let csv3 = "k,Neff_A,Neff_B,mismatch,se_A,se_B\n";
for (const d of fig3Data) {
  csv3 += `${d.k},${d.complexA.toFixed(3)},${d.complexB.toFixed(3)},${d.mismatch.toFixed(4)},${d.seA.toFixed(3)},${d.seB.toFixed(3)}\n`;
}
fs.writeFileSync('fig3_complexity_vs_k.csv', csv3);
console.log("Wrote fig3_complexity_vs_k.csv");

// Figure 4 data
let csv4 = "lambda,Neff_B,mismatch\n";
for (const d of fig4Data) {
  csv4 += `${d.lambda},${d.complexB.toFixed(3)},${d.mismatch.toFixed(4)}\n`;
}
fs.writeFileSync('fig4_complexity_vs_lambda.csv', csv4);
console.log("Wrote fig4_complexity_vs_lambda.csv");

// Random projection control data
let csvRandom = "k,Neff_A,Neff_B,mismatch,se_A,se_B\n";
for (const d of randomProjData) {
  csvRandom += `${d.k},${d.complexA.toFixed(3)},${d.complexB.toFixed(3)},${d.mismatch.toFixed(4)},${d.seA.toFixed(3)},${d.seB.toFixed(3)}\n`;
}
fs.writeFileSync('fig_random_projection.csv', csvRandom);
console.log("Wrote fig_random_projection.csv");

// =============================================================================
// OUTPUT LATEX TABLE
// =============================================================================

console.log("\n=== LATEX TABLE (for Results section) ===\n");

console.log("\\begin{table}[h]");
console.log("\\centering");
console.log("\\caption{Effective dimensionality and mismatch as a function of code bandwidth $k$.}");
console.log("\\label{tab:results}");
console.log("\\begin{tabular}{ccccc}");
console.log("\\toprule");
console.log("$k$ & $\\Neff(A)$ & $\\Neff(B)$ & Mismatch ($\\Delta$) \\\\");
console.log("\\midrule");
for (const d of fig3Data) {
  console.log(`${d.k} & ${d.complexA.toFixed(1)} $\\pm$ ${d.seA.toFixed(1)} & ${d.complexB.toFixed(1)} $\\pm$ ${d.seB.toFixed(1)} & ${d.mismatch.toFixed(3)} \\\\`);
}
console.log("\\bottomrule");
console.log("\\end{tabular}");
console.log("\\end{table}");

// =============================================================================
// SUMMARY STATISTICS FOR PAPER TEXT
// =============================================================================

console.log("\n=== PAPER TEXT STATISTICS ===\n");

const lowK = fig3Data.find(d => d.k === 1);
const highK = fig3Data.find(d => d.k === 32);
const meanA = fig3Data.reduce((s, d) => s + d.complexA, 0) / fig3Data.length;

console.log(`At k=1: N_eff(B) = ${lowK.complexB.toFixed(1)} ± ${lowK.seB.toFixed(1)}`);
console.log(`At k=32: N_eff(B) = ${highK.complexB.toFixed(1)} ± ${highK.seB.toFixed(1)}`);
console.log(`Mean N_eff(A) across all k: ${meanA.toFixed(1)}`);
console.log(`Mismatch at k=1: ${lowK.mismatch.toFixed(3)}`);
console.log(`Mismatch at k=32: ${highK.mismatch.toFixed(3)}`);
console.log(`Control mismatch (uncoupled): ${(controlMismatch / nTrials).toFixed(3)}`);
