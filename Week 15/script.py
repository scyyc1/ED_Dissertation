# Packages
import numpy as np
import matplotlib.pyplot as plt

# Self-defined functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from util.mesh.triangle.algorithm.R2.mapping_poly_square import Chen_2023_scipy

from util.pyvista_util import preprocess, postprocess

tritess_v, tritess_f = preprocess("./mesh/cat/cat_input.ply")
tritess_test = Chen_2023_scipy(tritess_v, tritess_f)
tritess_test.optimize_default()
tritess_test.mapping()
postprocess("./mesh/cat/cat_output2.ply", tritess_test.solution, tritess_f)