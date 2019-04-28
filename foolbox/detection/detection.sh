# Vanilla Boundary for ~200,000 queries
PYTHONPATH=~/foolbox python detection_boundary_attack.py --num_iters 12500 --normal_factor 1.0 --idx-range 0 50 --save-name detection_normal_12_5k_s0_e50.npz

# Brightness Transform
PYTHONPATH=~/foolbox python detection_boundary_attack.py --num_iters 100000 --normal_factor 1.0 --transform brightness --transform_param 0.18 --idx-range 0 100 --save-name detection_perlin_brightness_12_5k.npz

# Scale Transform
PYTHONPATH=~/foolbox python detection_boundary_attack.py --num_iters 100000 --normal_factor 1.0 --transform pixel_scale --transform_param 0.34 --idx-range 0 100 --save-name detection_perlin_scale_12_5k.npz

# Contrast Transform
PYTHONPATH=~/foolbox python detection_boundary_attack.py --num_iters 100000 --normal_factor 1.0 --transform contrast --transform_param 0.77 --idx-range 0 100 --save-name detection_perlin_contrast_12_5k.np

