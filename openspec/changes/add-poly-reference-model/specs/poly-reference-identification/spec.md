## ADDED Requirements

### Requirement: Poly-reference FRF array

`ModelPoly` SHALL accept an FRF array of shape `(n_outputs, n_freq, n_inputs)`
together with a frequency vector of length `n_freq`, and SHALL expose
`n_outputs` and `n_inputs`. A 2-D FRF array of shape `(n_outputs, n_freq)` MUST
be treated as a single input (`n_inputs == 1`).

#### Scenario: 3-D FRF array
- **WHEN** `ModelPoly` is created with an FRF array of shape `(6, 1000, 2)` and a frequency vector of length 1000
- **THEN** `n_outputs` is 6, `n_inputs` is 2 and `frf` has shape `(6, 1000, 2)`

#### Scenario: 2-D FRF array is a single input
- **WHEN** `ModelPoly` is created with an FRF array of shape `(6, 1000)`
- **THEN** `n_inputs` is 1 and `frf` has shape `(6, 1000, 1)`

### Requirement: Invalid input is rejected

`ModelPoly` MUST raise `ValueError` when the FRF array is neither 2-D nor 3-D,
when its frequency axis does not match the length of the frequency vector, or
when the frequency limits do not satisfy `freq_lower < freq_upper` inside the
frequency vector's range.

#### Scenario: Frequency vector length mismatch
- **WHEN** `ModelPoly` is created with an FRF array of shape `(6, 1000, 2)` and a frequency vector of length 999
- **THEN** `ValueError` is raised

#### Scenario: 4-D FRF array
- **WHEN** `ModelPoly` is created with a 4-D FRF array
- **THEN** `ValueError` is raised

### Requirement: Poles and participation factors per model order

`get_poles()` SHALL estimate poles with the poly-reference least-squares complex
frequency method for every model order from 1 to `pol_order_high`. For each
order it SHALL store the poles with positive imaginary part in `all_poles`,
their natural frequencies in `pole_freq`, their damping ratios in
`pole_damping_ratio` and their participation factors, an array of shape
`(n_inputs, n_poles)`, in `pole_participation_factor`.

#### Scenario: One entry per model order
- **WHEN** `get_poles()` runs on a model with `pol_order_high = 20` and `n_inputs = 2`
- **THEN** `all_poles`, `pole_freq`, `pole_damping_ratio` and `pole_participation_factor` each have 20 entries
- **AND** every entry of `pole_participation_factor` has 2 rows and as many columns as the matching entry of `all_poles` has poles

### Requirement: Pole selection by approximate natural frequency

`select_closest_poles(approx_natural_freq)` SHALL select, for each approximate
natural frequency, the closest stable pole and SHALL store `natural_freq`,
`damping_ratio`, `poles`, `pole_idx` and `participation_factor` (shape
`(n_inputs, n_modes)`) in the order of the given frequencies. It MUST raise
`ValueError` when two approximate frequencies resolve to the same pole.

#### Scenario: Selected modes follow the given order
- **WHEN** `select_closest_poles` is called with the six natural frequencies of the synthetic model
- **THEN** `natural_freq` and `damping_ratio` have length 6 and `participation_factor` has shape `(n_inputs, 6)`

#### Scenario: Same pole twice
- **WHEN** `select_closest_poles` is called with two equal approximate frequencies
- **THEN** `ValueError` is raised

### Requirement: Modal constants with participation factors

`get_constants()` SHALL fit, by least squares over the frequency band, the mode
shapes of the selected poles with their participation factors held fixed,
together with a lower and an upper residual for every output-input pair. It
SHALL store `mode_shape` (shape `(n_outputs, n_modes)`), `residual_lower` and
`residual_upper` (shape `(n_outputs, n_inputs)`) and `frf_reconstructed` (shape
of `frf`).

#### Scenario: Shapes of the modal constants
- **WHEN** `get_constants()` runs after selecting 6 modes on a model with 6 outputs and 2 inputs
- **THEN** `mode_shape` has shape `(6, 6)`, `residual_lower` and `residual_upper` have shape `(6, 2)` and `frf_reconstructed` has the shape of `frf`

### Requirement: Driving-point scaling

When `driving_point` gives, for every input, the output index measured at the
excitation point, `get_constants()` SHALL scale `mode_shape` and
`participation_factor` so that for every mode the participation factor equals
the mode shape at the driving points in the least-squares sense (unit modal A
scaling). The residues `mode_shape[:, r] * participation_factor[:, r]` MUST NOT
change by this scaling.

#### Scenario: A-normalized mode shapes
- **WHEN** the synthetic poly-reference model is identified with `driving_point` set
- **THEN** each identified mode shape is within 5 % (relative norm of the difference) of the model's A-normalized eigenvector, up to sign

### Requirement: Accuracy on noisy synthetic poly-reference data

On FRFs estimated by a multiple-input H1 estimator from simulated periodic
random excitation at two DOFs of a lumped model, with 2 % rms noise added to
every measured excitation and response signal, the identified natural
frequencies SHALL be within 0.1 % and the damping ratios within 2 % of the
model's values, and the MAC between each identified mode shape and the model's
eigenvector, and between each participation factor and the eigenvector at the
excitation DOFs, SHALL be at least 0.995.

#### Scenario: Noisy MIMO identification
- **WHEN** the noisy synthetic poly-reference data set is identified with fixed random seed
- **THEN** natural frequencies, damping ratios, mode shapes and participation factors meet the stated tolerances for all modes

### Requirement: Closely spaced modes are separated

`ModelPoly` SHALL identify both modes of a pair whose frequency spacing is less
than 10 % of their half-power bandwidth as two distinct poles, each within the
tolerances of the noisy synthetic data set.

#### Scenario: Close pair from two weakly coupled sub-systems
- **WHEN** the synthetic model of two equal, weakly coupled sub-systems is identified from the noisy data set
- **THEN** both modes of its closest pair are selected as distinct poles and meet the stated tolerances

### Requirement: Exact data is fitted exactly

On noise-free FRFs computed directly from the lumped model, the identified
natural frequencies SHALL be within 1e-6 and the damping ratios within 1e-5
(relative) of the model's values, and the relative error of
`frf_reconstructed` in the frequency band SHALL be below 1e-5.

#### Scenario: Noise-free MIMO identification
- **WHEN** the exact FRFs of the synthetic model at two references are identified
- **THEN** natural frequencies, damping ratios and the reconstructed FRF meet the stated tolerances

### Requirement: Single-reference data

On single-reference FRFs, `ModelPoly` SHALL identify the natural frequencies
and damping ratios of the synthetic single-reference test model within the
tolerances the test suite applies to `Model` on the same data.

#### Scenario: Single-reference synthetic FRF
- **WHEN** `ModelPoly` identifies the 2-D FRF of the existing single-reference synthetic test helper
- **THEN** natural frequencies and damping ratios pass the same closeness criteria as `Model` does on that data

### Requirement: Public import

`ModelPoly` SHALL be importable as `from sdypy.EMA import ModelPoly`.

#### Scenario: Import
- **WHEN** `from sdypy.EMA import ModelPoly` is executed
- **THEN** the import succeeds and `ModelPoly` is a class
