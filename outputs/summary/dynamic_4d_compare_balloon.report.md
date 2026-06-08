# Dynamic 4D Modeling Comparison

## Methods
- KalmanPrevious: `output\Bonn\bonn_balloon\4d_model_previous_motion\model_4d.npz`
- LearnedRegion: `outputs\Bonn\bonn_balloon\4d_model_learned_motion_region\model_4d.npz`

## Key Metrics (Your Target)
- `motion_nn_mean_m` (lower is better): nan -> 0.075402 m
- `motion_nn_median_m` (lower is better): nan -> 0.046474 m
- `motion_vs_static_gain_percent` (higher is better): +nan% -> +2.90%
- `motion_inlier_ratio` (higher is better): 0.0000 -> 0.9800
- `axis_bias_l2_m` (lower is better): 0.000000 -> 0.001615 m

## Relative Change (Baseline -> Learned)
- motion_nn_mean gain: +nan%
- motion_nn_median gain: +nan%
- motion_vs_static_gain delta: +nan%
- motion_inlier_ratio delta: +0.9800
- axis_bias_l2 gain: -161458855850.39%
