from fem_solver import Mesh, Material, Solver

def main():

    # ============================================ GEOMETRY ============================================================

    NUM_X = 21       # Number of nodes at x-coordinate
    NUM_Y = 6        # Number of nodes at y-coordinate
    LENGTH = 1.0     # Length along x-coordinate (m)
    HEIGHT = 0.25    # Height along y-coordinate (m)
    THICKNESS = 0.2  # Thickness of the plate (m)

    # ======================================== MATERIAL PROPERTIES =====================================================

    ELASTICITY_MODULUS = 206E9  # Young's Modulus (Pa)
    POISSON_RATIO = 0.3     # Poisson's Ration (-)
    PROBLEM_TYPE = "Plane Stress"  # Type of Problem: "Plane Strain" or "Plane Stress"

    # =============================================== MESH =============================================================

    mesh = Mesh()   # Initialize Mesh
    mesh.create_rectangular_mesh(   # Create rectangular mesh
        num_x=NUM_X,
        num_y=NUM_Y,
        length=LENGTH,
        height=HEIGHT,
        thickness=THICKNESS
    )
    mesh.plot_mesh()    # Plot Mesh

    # ============================================== MATERIAL ==========================================================

    material = Material(
        name="Steel",
        E=ELASTICITY_MODULUS,
        poisson_ratio=POISSON_RATIO,
        stress_state=PROBLEM_TYPE
    )

    # =============================================== SOLVER ===========================================================

    solver = Solver(mesh, material)
    solver.assemble_global_stiffness()

    # ========================================== BOUNDARY CONDITIONS ===================================================

    boundary_conditions = {node.node_id: ("ALL", 0.0) for node in mesh.nodes if node.coordinates['X'] == 0.0}
    solver.apply_boundary_conditions(boundary_conditions)

    # ================================================ FORCES ==========================================================

    max_x = max(node.coordinates['X'] for node in mesh.nodes)  # Maximum x-coordinate
    force_nodes = [node.node_id for node in mesh.nodes if node.coordinates['X'] == max_x]
    for node_id in force_nodes:
        solver.apply_force(node_id=node_id, direction="Y", force_value=-10E5)
    solver.apply_force(node_id=20, direction="Y", force_value=10E5 / 2)
    solver.apply_force(node_id=125, direction="Y", force_value=10E5 / 2)

    # ================================================= SOLVE ==========================================================

    U = solver.solve()
    solver.plot_deformation(scale=10.0)
    solver.plot_stress(stress_type="svm")

    print(solver)

    help(solver)

if __name__ == '__main__':
    main()

