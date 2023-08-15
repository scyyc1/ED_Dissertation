# Packages
import time

# Self-defined functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from util.mesh.triangle.algorithm.R2.my_test import boundary_smoothing_tutte, poly_square_tutte

from util.pyvista_util import preprocess, postprocess

tritess_v, tritess_f = preprocess("./mesh/cat/cat_input.ply")

# Poly square mapping
tritess_test1 = poly_square_tutte(tritess_v.copy(), tritess_f.copy())

# Boundary Deformation
start_time = time.time()
tritess_test1.optimize_default()
end_time = time.time()

t_boundary_deformation1 = end_time - start_time

# Inner vertices deformation
start_time = time.time()
tritess_test1.mapping()
end_time = time.time()

t_tutte1 = end_time - start_time

postprocess("./mesh/cat/cat_out_PolySquare.ply", tritess_test1.solution, tritess_f)

# Boundary smoothing
tritess_test2 = boundary_smoothing_tutte(tritess_v.copy(), tritess_f.copy())

# Boundary Deformation
start_time = time.time()
tritess_test2.optimize_default()
end_time = time.time()

t_boundary_deformation2 = end_time - start_time

# Inner vertices deformation
start_time = time.time()
tritess_test2.mapping()
end_time = time.time()

t_tutte2 = end_time - start_time

postprocess("./mesh/cat/cat_out_Smoothing.ply", tritess_test2.solution, tritess_f)