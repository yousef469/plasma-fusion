"""
V3 Tokamak 3D Model Generator
Generates STL mesh + matplotlib visualization of the final V3 design.
Pure Python (numpy only) — no CAD library needed.
"""

import math
import numpy as np

# =============================================================================
# V3 Design Parameters
# =============================================================================
R0 = 7.0       # major radius (m)
a = 1.2        # minor radius (m)
kappa = 3.0    # elongation
delta = -0.30  # triangularity (negative)
N_coil = 16    # TF coils
r_blk = 0.80   # blanket thickness (m)
r_shd = 0.50   # shield thickness
r_vv = 0.30    # vacuum vessel
r_gap = 0.20   # gap
inboard_build = r_blk + r_shd + r_vv + r_gap
R_TF_inner = R0 - a - inboard_build  # TF inner leg radius


# =============================================================================
# Geometry Utilities
# =============================================================================
def plasma_contour(theta, R0=R0, a=a, kappa=kappa, delta=delta):
    """
    Poloidal plasma boundary in (R, Z) coordinates.
    theta: poloidal angle (0 to 2pi)
    Returns (R, Z) arrays of the shaped plasma boundary.
    """
    r = a * np.ones_like(theta)
    R = R0 + r * np.cos(theta + delta * np.sin(theta))
    Z = kappa * r * np.sin(theta)
    return R, Z


def tf_coil_shape(R0=R0, a=a, R_inner=R_TF_inner, coil_w=0.6, coil_th=0.3):
    """
    Generate D-shaped TF coil outline.
    Returns (R, Z) coordinates of the coil centerline.
    """
    N = 50
    theta = np.linspace(0, np.pi, N)

    # Inner leg: straight vertical section at R_inner
    h_leg = 2.0 * (a + 0.5) * 1.2 * 1.2  # coil height
    R_in = R_inner * np.ones(N)
    Z_in = np.linspace(-h_leg/2, h_leg/2, N)

    # Outer leg: D-curve from top to bottom
    theta_arc = np.linspace(0, np.pi, N)
    R_out = R_inner + (R0 + a * 1.5 - R_inner) * np.sin(theta_arc)
    Z_out = h_leg/2 * np.cos(theta_arc)

    # Full coil: inner leg + top arc + outer leg + bottom arc
    R_coil = np.concatenate([R_in, R_out[::-1]])
    Z_coil = np.concatenate([Z_in, Z_out[::-1]])

    return R_coil, Z_coil


def toroidal_coil_shape(R_coil, Z_coil, phi, R0=R0):
    """
    Rotate a poloidal coil shape to toroidal angle phi.
    """
    R_3d = R_coil * np.cos(phi)
    Z_3d = Z_coil
    # The coil is at toroidal angle phi: rotate around Z axis
    X = R_coil * np.cos(phi)
    Y = R_coil * np.sin(phi)
    Z = Z_coil
    return X, Y, Z


# =============================================================================
# STL Writer (pure Python, no dependencies)
# =============================================================================
def write_stl(filename, vertices, triangles, name="V3 Tokamak"):
    """
    Write binary STL file.
    vertices: (N, 3) array of xyz coordinates
    triangles: (M, 3) array of vertex indices
    """
    import struct
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        f.write(name.encode().ljust(80, b'\x00'))
        # Number of triangles (4 bytes, little-endian unsigned int)
        f.write(struct.pack('<I', len(triangles)))
        # Each triangle: normal (12 bytes) + 3 vertices (36 bytes) + attribute (2 bytes)
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])
            f.write(struct.pack('<3f', normal[0], normal[1], normal[2]))
            for v in [v0, v1, v2]:
                f.write(struct.pack('<3f', float(v[0]), float(v[1]), float(v[2])))
            f.write(struct.pack('<H', 0))  # attribute byte count
    print(f"  Wrote {len(triangles)} triangles to {filename}")


def create_torus_mesh(R0, a, kappa=1.0, delta=0.0,
                      n_phi=48, n_theta=32, scale=1.0):
    """
    Create triangular mesh for a shaped torus.
    Uses consistent parameterization: R = R0 + a*cos(θ + δ sin θ), Z = κ*a*sin θ
    """
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

    vertices = []
    for j, p in enumerate(phi):
        for i, t in enumerate(theta):
            R_pt = R0 + a * np.cos(t + delta * np.sin(t))
            Z_pt = kappa * a * np.sin(t)
            vertices.append([R_pt * np.cos(p) * scale,
                             R_pt * np.sin(p) * scale,
                             Z_pt * scale])

    vertices = np.array(vertices)

    # Generate triangles (two per quadrilateral)
    triangles = []
    for j in range(n_phi):
        for i in range(n_theta):
            j1 = (j + 1) % n_phi
            i1 = (i + 1) % n_theta
            v0 = j * n_theta + i
            v1 = j * n_theta + i1
            v2 = j1 * n_theta + i
            v3 = j1 * n_theta + i1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    return vertices, np.array(triangles)


def create_tf_coil_mesh(R0, R_inner, a, N_coil=16, n_phi=48,
                        coil_w=0.5, scale=1.0):
    """
    Create triangular mesh for TF coils.
    Returns combined (vertices, triangles) for all coils.
    """
    h_leg = 2.0 * (a + 0.5) * 1.2 * 1.2
    R_outer = R0 + a * 1.5

    all_vertices = []
    all_triangles = []
    offset = 0

    phi_coils = np.linspace(0, 2*np.pi, N_coil, endpoint=False)

    for phi_c in phi_coils:
        # Generate rectangular cross-section coil path
        n_seg = 32
        theta = np.linspace(-np.pi/2, np.pi/2, n_seg)
        # D-shape: inner straight + outer arc
        R_pts = []
        Z_pts = []
        # Inner leg (straight)
        for z in np.linspace(-h_leg/2, h_leg/2, n_seg//2):
            R_pts.append(R_inner)
            Z_pts.append(z)
        # Outer leg (arc)
        for t in np.linspace(0, np.pi, n_seg//2):
            frac = t / np.pi
            R = R_inner + (R_outer - R_inner) * np.sin(t)
            Z = h_leg/2 * np.cos(t)
            R_pts.append(R)
            Z_pts.append(z)

        # Create coil as extruded rectangle along the path
        coil_verts = []
        for i in range(n_seg):
            r = R_pts[i]
            z = Z_pts[i]
            # Coil cross-section: rectangle with corners at ±coil_w/2
            for dx, dz in [(-coil_w/2, -coil_w/2), (coil_w/2, -coil_w/2),
                           (coil_w/2, coil_w/2), (-coil_w/2, coil_w/2)]:
                x = (r + dx) * np.cos(phi_c)
                y = (r + dx) * np.sin(phi_c)
                z_out = z + dz
                coil_verts.append([x * scale, y * scale, z_out * scale])

        coil_verts = np.array(coil_verts)
        all_vertices.append(coil_verts)

        # Connect rectangles along the path
        for i in range(n_seg):
            i_next = (i + 1) % n_seg
            base = offset + i * 4
            base_next = offset + i_next * 4
            # 6 triangles per quad between adjacent cross-sections
            tris = [
                [base, base+1, base_next],
                [base+1, base_next+1, base_next],
                [base+1, base+2, base_next+1],
                [base+2, base_next+2, base_next+1],
                [base+2, base+3, base_next+2],
                [base+3, base_next+3, base_next+2],
                [base+3, base, base_next+3],
                [base, base_next, base_next+3],
                # End caps at first and last segment
            ]
            if i == 0:
                tris.extend([
                    [base, base+3, base+2],
                    [base, base+2, base+1],
                ])
            if i == n_seg - 1:
                tris.extend([
                    [base_next, base_next+1, base_next+2],
                    [base_next, base_next+2, base_next+3],
                ])
            all_triangles.extend(tris)
        offset += len(coil_verts)

    all_vertices = np.vstack(all_vertices) if all_vertices else np.array([])
    return all_vertices, np.array(all_triangles)


def create_pf_coil_mesh(R0, a, kappa, N_pf=6, scale=1.0):
    """Create PF coil rings at various positions."""
    pf_positions = [
        (R0 + a * 0.8, a * kappa * 0.9),   # upper divertor
        (R0 - a * 0.5, a * kappa * 0.8),   # upper shaping
        (R0 + a * 1.5, a * kappa * 0.6),   # upper outer
        (R0 + a * 1.5, -a * kappa * 0.6),  # lower outer
        (R0 - a * 0.5, -a * kappa * 0.8),  # lower shaping
        (R0 + a * 0.8, -a * kappa * 0.9),  # lower divertor
    ]

    all_vertices = []
    all_triangles = []
    offset = 0
    n_seg = 24
    wire_r = 0.15  # coil wire radius
    n_circ = 8     # circumferential segments

    for R_pf, Z_pf in pf_positions:
        phi = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
        for p in phi:
            center = np.array([R_pf * np.cos(p), R_pf * np.sin(p), Z_pf])
            # Local basis for toroidal direction
            t_dir = np.array([-np.sin(p), np.cos(p), 0.0])
            # Normal pointing radially outward
            n_dir = np.array([np.cos(p), np.sin(p), 0.0])
            b_dir = np.array([0, 0, 1.0])

            circ_theta = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
            for ct in circ_theta:
                x = center + wire_r * (np.cos(ct) * n_dir + np.sin(ct) * b_dir)
                all_vertices.append(x * scale)

        n_this = n_seg * n_circ
        for i in range(n_seg):
            for j in range(n_circ):
                i1 = (i + 1) % n_seg
                j1 = (j + 1) % n_circ
                v0 = offset + i * n_circ + j
                v1 = offset + i * n_circ + j1
                v2 = offset + i1 * n_circ + j
                v3 = offset + i1 * n_circ + j1
                all_triangles.extend([[v0, v1, v2], [v1, v3, v2]])

        offset += n_this

    return np.array(all_vertices), np.array(all_triangles)


def create_cs_mesh(R0, R_inner, scale=1.0):
    """Create central solenoid as a cylinder."""
    R_cs = min(R_inner * 0.35, 1.5)
    h_cs = 8.0
    n_phi = 24
    n_h = 12

    verts = []
    for j in range(n_phi):
        p = 2*np.pi * j / n_phi
        for k in range(n_h + 1):
            z = -h_cs/2 + h_cs * k / n_h
            # Inner surface
            verts.append([(R_cs - 0.2) * np.cos(p) * scale,
                          (R_cs - 0.2) * np.sin(p) * scale, z * scale])
            # Outer surface
            verts.append([(R_cs + 0.2) * np.cos(p) * scale,
                          (R_cs + 0.2) * np.sin(p) * scale, z * scale])

    verts = np.array(verts)
    tris = []
    n_per_layer = 2 * n_phi

    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        for k in range(n_h):
            v0 = k * n_per_layer + 2*j
            v1 = k * n_per_layer + 2*j + 1
            v2 = (k+1) * n_per_layer + 2*j
            v3 = (k+1) * n_per_layer + 2*j + 1
            v4 = k * n_per_layer + 2*j1
            v5 = k * n_per_layer + 2*j1 + 1
            v6 = (k+1) * n_per_layer + 2*j1
            v7 = (k+1) * n_per_layer + 2*j1 + 1

            # Inner wall
            tris.extend([[v0, v4, v2], [v4, v6, v2]])
            # Outer wall
            tris.extend([[v1, v3, v5], [v5, v3, v7]])
            # Top/bottom caps
            if k == 0:
                tris.extend([[v0, v1, v5], [v0, v5, v4]])
            if k == n_h - 1:
                tris.extend([[v2, v6, v7], [v2, v7, v3]])

    return verts, np.array(tris)


# =============================================================================
# Main: Generate all components and export
# =============================================================================
def main():
    scale = 1.0  # meters

    print("Generating V3 tokamak 3D model...")
    print(f"  R0={R0}m, a={a}m, k={kappa}, N_coil={N_coil}")
    print(f"  R_TF_inner={R_TF_inner:.1f}m, B_peak=22.8T")
    print()

    # 1. Plasma
    print("1. Plasma torus...")
    plasma_vert, plasma_tri = create_torus_mesh(
        R0, a, kappa, delta, n_phi=64, n_theta=48, scale=scale)
    write_stl("/tmp/v3_plasma.stl", plasma_vert, plasma_tri, "V3 Plasma")
    print(f"   Vertices: {len(plasma_vert)}, Triangles: {len(plasma_tri)}")

    # 2. TF coils
    print("\n2. TF coils (16 D-shaped)...")
    tf_vert, tf_tri = create_tf_coil_mesh(
        R0, R_TF_inner, a, N_coil=16, coil_w=0.5, scale=scale)
    write_stl("/tmp/v3_tf_coils.stl", tf_vert, tf_tri, "V3 TF Coils")
    print(f"   Vertices: {len(tf_vert)}, Triangles: {len(tf_tri)}")

    # 3. PF coils
    print("\n3. PF coils (6 rings)...")
    pf_vert, pf_tri = create_pf_coil_mesh(R0, a, kappa, N_pf=6, scale=scale)
    write_stl("/tmp/v3_pf_coils.stl", pf_vert, pf_tri, "V3 PF Coils")
    print(f"   Vertices: {len(pf_vert)}, Triangles: {len(pf_tri)}")

    # 4. Central solenoid
    print("\n4. Central solenoid...")
    cs_vert, cs_tri = create_cs_mesh(R0, R_TF_inner, scale=scale)
    write_stl("/tmp/v3_cs.stl", cs_vert, cs_tri, "V3 Central Solenoid")
    print(f"   Vertices: {len(cs_vert)}, Triangles: {len(cs_tri)}")

    # 5. Combined model
    print("\n5. Combined model...")
    all_vert = [plasma_vert, tf_vert, pf_vert, cs_vert]
    all_tri = [plasma_tri, tf_tri, pf_tri, cs_tri]

    combined_vert = np.vstack(all_vert)
    offset = 0
    combined_tri = []
    for i, (vert, tri) in enumerate(zip(all_vert, all_tri)):
        combined_tri.append(tri + offset)
        offset += len(vert)
    combined_tri = np.vstack(combined_tri)
    write_stl("/tmp/v3_tokamak.stl", combined_vert, combined_tri, "V3 Tokamak")
    print(f"   Total vertices: {len(combined_vert)}, Triangles: {len(combined_tri)}")

    # 6. Matplotlib visualization (2D cross-section)
    print("\n6. Generating 2D cross-section...")
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 2D poloidal cross-section
        theta = np.linspace(0, 2*np.pi, 200)
        R_plas, Z_plas = plasma_contour(theta, R0, a, kappa, delta)
        ax1.plot(R_plas, Z_plas, 'b-', lw=2, label='Plasma')

        # TF coil outline
        R_coil, Z_coil = tf_coil_shape()
        ax1.plot(R_coil, Z_coil, 'r-', lw=1.5, label='TF coil')

        # PF coils
        pf_pos = [(R0 + a*0.8, a*kappa*0.9), (R0 - a*0.5, a*kappa*0.8),
                  (R0 + a*1.5, a*kappa*0.6), (R0 + a*1.5, -a*kappa*0.6),
                  (R0 - a*0.5, -a*kappa*0.8), (R0 + a*0.8, -a*kappa*0.9)]
        for r, z in pf_pos:
            ax1.plot(r, z, 'gs', ms=8)

        # CS
        ax1.plot([0, 0], [-4, 4], 'k-', lw=3, label='CS')

        ax1.set_xlabel('R (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('V3 Poloidal Cross-Section')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # 3D view
        ax2 = fig.add_subplot(122, projection='3d')
        n_phi_viz = 48
        n_theta_viz = 32
        phi_viz = np.linspace(0, 2*np.pi, n_phi_viz)
        theta_viz = np.linspace(0, 2*np.pi, n_theta_viz)
        PHI, THETA = np.meshgrid(phi_viz, theta_viz)
        R_viz = R0 + a * np.cos(THETA + delta * np.sin(THETA))
        Z_viz = kappa * a * np.sin(THETA)
        X_viz = R_viz * np.cos(PHI)
        Y_viz = R_viz * np.sin(PHI)
        ax2.plot_surface(X_viz, Y_viz, Z_viz, alpha=0.6, color='b')

        # TF coils (simplified)
        phi_coils = np.linspace(0, 2*np.pi, N_coil, endpoint=False)
        for pc in phi_coils:
            R_c, Z_c = tf_coil_shape()
            X_c = R_c * np.cos(pc)
            Y_c = R_c * np.sin(pc)
            ax2.plot(X_c, Y_c, Z_c, 'r-', lw=1, alpha=0.5)

        # PF coils (rings)
        for r, z in pf_pos:
            phi_pf = np.linspace(0, 2*np.pi, 32)
            X_pf = r * np.cos(phi_pf)
            Y_pf = r * np.sin(phi_pf)
            Z_pf = z * np.ones_like(phi_pf)
            ax2.plot(X_pf, Y_pf, Z_pf, 'g-', lw=2)

        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('V3 3D View')
        ax2.set_box_aspect([1, 1, 0.6])

        plt.tight_layout()
        plt.savefig('/tmp/v3_tokamak.png', dpi=150)
        print("  Saved /tmp/v3_tokamak.png")
        plt.close()

    except ImportError:
        print("  matplotlib not available, skipping visualization")

    print("\nDone! STL files in /tmp/:")

    import os
    for f in sorted(os.listdir('/tmp/')):
        if f.startswith('v3_') and f.endswith('.stl'):
            sz = os.path.getsize(f'/tmp/{f}')
            print(f"  /tmp/{f} ({sz/1024:.0f} KB)")


if __name__ == "__main__":
    main()
