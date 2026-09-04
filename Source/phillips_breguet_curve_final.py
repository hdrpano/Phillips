
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, brentq
from scipy.integrate import quad
from scipy.spatial import cKDTree


@dataclass
class SpringParameters:
    inner_diameter: float = 0.55
    spring_thickness: float = 0.034
    spring_height: float = 0.120
    turns: float = 13.0
    winding_factor: float = 4.0

    attachment_angle_deg: float = 110.0
    curve_ratio: float = 0.70
    beta_deg: float = 240.0

    lift_angle_deg: float = 60.0
    lift_height_factor: float = 3.0
    kick_radius: float = 0.120          # mm, symmetric start/end bend radius

    piton_extra_deg: float = 5.0

    unbent_z: float = -1.0
    lower_z: float = 0.0

    exact_points_per_turn: int = 180
    terminal_exact_points: int = 800

    fusion_tolerance_mm: float = 0.0010
    fusion_max_points: int = 500
    fusion_points_per_turn: int = 16
    fusion_lift_points: int = 25
    fusion_terminal_points: int = 70


def derive(p):
    # Winding rule:
    # winding_factor * spring_thickness is the RADIAL pitch per turn.
    # Therefore the diameter grows by twice that value per turn.
    rp = p.winding_factor * p.spring_thickness
    dp = 2.0 * rp
    ri = p.inner_diameter / 2.0

    nominal_or = ri + p.turns * rp
    nominal_od = 2.0 * nominal_or

    inner_angle = p.beta_deg - p.attachment_angle_deg
    lower_extent_deg = p.turns * 360.0 + p.attachment_angle_deg - p.beta_deg
    lower_turns = lower_extent_deg / 360.0

    rho0 = ri + rp * lower_turns
    curve_diam = p.curve_ratio * nominal_od
    d = curve_diam / 2.0

    lift_height = p.lift_height_factor * p.spring_height

    return dict(
        diametral_pitch=dp,
        radial_pitch=rp,
        free_gap=rp - p.spring_thickness,
        inner_radius=ri,
        nominal_outer_diameter=nominal_od,
        nominal_outer_radius=nominal_or,
        inner_attachment_angle_deg=inner_angle,
        lower_extent_deg=lower_extent_deg,
        lower_turns_effective=lower_turns,
        rho0=rho0,
        curve_inner_diameter=curve_diam,
        curve_radius=d,
        phillips_number=100.0*d/rho0,
        lift_height=lift_height,
        upper_z=p.lower_z + lift_height,
        lift_start_deg=lower_extent_deg - p.lift_angle_deg,
        total_final_angle_deg=p.turns*360.0 + p.attachment_angle_deg,
    )


def r_arch(theta, p, g):
    return g["inner_radius"] + g["radial_pitch"] * np.asarray(theta) / (2*math.pi)


def xy_arch(theta, p, g):
    phi0 = math.radians(g["inner_attachment_angle_deg"])
    r = r_arch(theta, p, g)
    phi = phi0 + theta
    return r*np.cos(phi), r*np.sin(phi)


def solve_lift_profile(planar_length, height, radius):
    """
    Symmetric 3D lift in the developed s-z plane:

        circular kick R -> straight tangent -> circular counter-kick R

    The projected planar length is fixed by the 60° Archimedean section.
    The end tangents are horizontal at both z levels.
    """
    if radius <= 0.0:
        raise ValueError("kick_radius must be > 0.")
    if height <= 0.0:
        raise ValueError("lift height must be > 0.")

    def residual(alpha):
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        lm = (planar_length - 2.0*radius*sa) / ca
        return (
            2.0*radius*(1.0-ca)
            + lm*sa
            - height
        )

    # Search for the first physically useful solution.
    lo = math.radians(0.01)
    hi = math.radians(80.0)

    alpha = brentq(residual, lo, hi)

    sa = math.sin(alpha)
    ca = math.cos(alpha)

    x_arc = radius * sa
    z_arc = radius * (1.0-ca)
    middle_length = (
        planar_length - 2.0*x_arc
    ) / ca

    if middle_length < 0.0:
        raise ValueError(
            "Kick radius is too large for the available lift length."
        )

    x_middle = middle_length * ca
    z_middle = middle_length * sa

    return dict(
        alpha=alpha,
        alpha_deg=math.degrees(alpha),
        radius=radius,
        x_arc=x_arc,
        z_arc=z_arc,
        middle_length=middle_length,
        x_middle=x_middle,
        z_middle=z_middle,
        planar_length=planar_length,
        height=height,
    )


def lift_z_from_planar_s(s, profile, z0=0.0):
    """
    Height z(s) for the symmetric kick-radius lift.
    s is the projected planar arc length from lift start.
    """
    s = np.asarray(s, dtype=float)

    R = profile["radius"]
    a = profile["alpha"]
    x1 = profile["x_arc"]
    x2 = x1 + profile["x_middle"]

    z1 = profile["z_arc"]
    z2 = z1 + profile["z_middle"]

    z = np.empty_like(s)

    m1 = s <= x1
    m2 = (s > x1) & (s < x2)
    m3 = s >= x2

    # First circular kick: tangent angle 0 -> alpha.
    if np.any(m1):
        q = np.arcsin(np.clip(s[m1] / R, -1.0, 1.0))
        z[m1] = z0 + R*(1.0-np.cos(q))

    # Straight tangent section.
    if np.any(m2):
        z[m2] = (
            z0 + z1
            + (s[m2]-x1)*math.tan(a)
        )

    # Counter-kick: tangent angle alpha -> 0.
    if np.any(m3):
        dx = s[m3]-x2
        sin_q = math.sin(a) - dx/R
        q = np.arcsin(np.clip(sin_q, -1.0, 1.0))
        z[m3] = (
            z0 + z2
            + R*(np.cos(q)-math.cos(a))
        )

    return z


def lower_finished_exact(p, g):
    """
    Finished lower Archimedean spring including the 60° 3D lift.

    The XY projection stays exactly on the Archimedean spiral.
    The final 60° are lifted with two small circular kick radii and a
    straight tangent section. Therefore the tangent is horizontal at the
    start and again horizontal at terminal birth D.
    """
    extent = math.radians(g["lower_extent_deg"])
    lift = math.radians(p.lift_angle_deg)
    lift_start = extent - lift

    n = int(math.ceil(
        g["lower_turns_effective"]*p.exact_points_per_turn
    )) + 1

    th = np.linspace(0.0, extent, n)
    x, y = xy_arch(th, p, g)
    z = np.full_like(th, p.lower_z)

    # Planar cumulative arc length of the exact Archimedean projection.
    xy = np.column_stack((x, y))
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s_all = np.r_[0.0, np.cumsum(ds)]

    lift_start_i = int(np.searchsorted(th, lift_start))
    lift_start_i = max(1, min(lift_start_i, len(th)-2))

    # Interpolate the exact planar s-value at lift_start.
    s_start = np.interp(lift_start, th, s_all)
    s_end = s_all[-1]
    planar_lift_length = s_end - s_start

    profile = solve_lift_profile(
        planar_lift_length,
        g["lift_height"],
        p.kick_radius,
    )

    m = th >= lift_start
    s_lift = s_all[m] - s_start
    s_lift = np.clip(
        s_lift,
        0.0,
        planar_lift_length,
    )

    z[m] = lift_z_from_planar_s(
        s_lift,
        profile,
        p.lower_z,
    )

    # Force exact endpoint to the upper level.
    z[-1] = g["upper_z"]

    return np.c_[x, y, z], th, profile


def terminal_from_params(p, g, v):
    beta = math.radians(p.beta_deg)
    uk = np.linspace(0,1,5)
    rk = np.array([g["rho0"], v[0], v[1], v[2], g["curve_radius"]])
    rs = CubicSpline(uk, rk, bc_type=((1,0.0),(1,0.0)))

    u = np.linspace(0,1,p.terminal_exact_points)
    r = rs(u)
    phi = beta*u + v[3]*np.sin(math.pi*u)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z = np.full_like(x, g["upper_z"])
    return np.c_[x,y,z]


def static_moments(points):
    xy = points[:,:2]
    d = np.diff(xy, axis=0)
    ds = np.linalg.norm(d, axis=1)
    mid = 0.5*(xy[:-1]+xy[1:])
    return float(ds.sum()), float(np.sum(mid[:,1]*ds)), float(np.sum(mid[:,0]*ds))


def optimise_terminal(p, g):
    rho0 = g["rho0"]
    d = g["curve_radius"]
    x0 = np.array([0.96*rho0,0.86*rho0,0.76*rho0,0.0])
    lo = min(rho0,d)*0.70
    hi = max(rho0,d)*1.30
    lb = np.array([lo,lo,lo,-0.70])
    ub = np.array([hi,hi,hi, 0.70])

    def fun(v):
        pts = terminal_from_params(p,g,v)
        _,mx,my = static_moments(pts)
        s = rho0*rho0
        rr = np.array([rho0,v[0],v[1],v[2],d])
        sm = np.diff(rr,n=2)/rho0
        return np.r_[7*(mx-s)/s, 7*my/s, 0.15*v[3], 0.07*sm]

    sol = least_squares(fun,x0,bounds=(lb,ub),max_nfev=3500,
                        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    pts = terminal_from_params(p,g,sol.x)
    L,mx,my = static_moments(pts)
    return pts, dict(
        success=bool(sol.success), length=L, mx=mx, my=my,
        target_mx=rho0*rho0,
        mx_error_pct=100*(mx-rho0*rho0)/(rho0*rho0),
        my_error_pct=100*my/(rho0*rho0),
    )


def length3d(points):
    return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())


def arch_length(theta,p,g):
    b = g["radial_pitch"]/(2*math.pi)
    ri = g["inner_radius"]
    f = lambda t: math.sqrt((ri+b*t)**2+b*b)
    return quad(f,0,theta,epsabs=1e-11,epsrel=1e-11,limit=100)[0]


def manufacturing_cut(p,g,finished):
    active = length3d(finished)
    f = lambda th: arch_length(th,p,g)-active
    hi = 2*math.pi*(p.turns+4)
    while f(hi)<0:
        hi *= 1.2
    ta = brentq(f,0,hi,xtol=1e-12,rtol=1e-12)
    tc = ta + math.radians(p.piton_extra_deg)
    return dict(
        active_length=active,
        theta_active=ta, theta_active_deg=math.degrees(ta),
        active_turns=ta/(2*math.pi),
        theta_cut=tc, theta_cut_deg=math.degrees(tc),
        cut_turns=tc/(2*math.pi),
    )


def unbent_exact(p,g,cut):
    n = int(math.ceil(cut["cut_turns"]*p.exact_points_per_turn))+1
    th = np.linspace(0,cut["theta_cut"],n)
    x,y = xy_arch(th,p,g)
    z = np.full_like(x,p.unbent_z)
    return np.c_[x,y,z]


# ---------- direct parametric Fusion fit points ----------

def chord_param(points):
    d = np.linalg.norm(np.diff(points,axis=0),axis=1)
    s = np.r_[0.0,np.cumsum(d)]
    if s[-1] == 0.0:
        return s
    return s/s[-1]


def reconstruct_dense_from_fit(fit, count=8000):
    """
    SciPy cubic spline used only as a proxy for Fusion SketchFittedSpline.
    The geometric error is evaluated by nearest distance, not by equal
    parameter values.
    """
    sf = chord_param(fit)
    keep = np.r_[True, np.diff(sf) > 1e-12]
    sf = sf[keep]
    fit = fit[keep]

    t = np.linspace(0.0,1.0,count)

    x = CubicSpline(sf,fit[:,0],bc_type="natural")(t)
    y = CubicSpline(sf,fit[:,1],bc_type="natural")(t)
    z = CubicSpline(sf,fit[:,2],bc_type="natural")(t)

    return np.c_[x,y,z]


def geometric_spline_error(exact, reconstructed):
    """
    Geometric reference-to-spline distance.

    Each high-resolution reference point is compared with the nearest point
    on a very densely sampled reconstructed spline.

    We deliberately do NOT use the reverse distance reconstructed -> exact:
    the exact reference is itself discretely sampled and that reverse
    distance would mostly measure the spacing of the reference samples,
    not the spline approximation error.
    """
    tree_rec = cKDTree(reconstructed)
    distance = tree_rec.query(exact, k=1)[0]
    return float(np.max(distance))


def interp_curve(parameter_exact, points_exact, parameter_new):
    return np.column_stack([
        np.interp(parameter_new, parameter_exact, points_exact[:,0]),
        np.interp(parameter_new, parameter_exact, points_exact[:,1]),
        np.interp(parameter_new, parameter_exact, points_exact[:,2]),
    ])


def build_finished_fusion_points(
    p,
    g,
    lower_exact,
    theta_lower,
    terminal_exact,
):
    """
    Generate Fusion fit points directly from the mathematical parameters.

    Flat Archimedean part:
        fusion_points_per_turn

    60° lift:
        fusion_lift_points

    Phillips terminal:
        fusion_terminal_points

    The lift start, lift end / D and terminal end are always preserved.
    """
    extent = math.radians(g["lower_extent_deg"])
    lift = math.radians(p.lift_angle_deg)
    lift_start = extent-lift

    # Flat section.
    flat_turns = lift_start/(2.0*math.pi)
    n_flat = max(
        4,
        int(math.ceil(
            flat_turns*p.fusion_points_per_turn
        )) + 1,
    )
    theta_flat = np.linspace(0.0,lift_start,n_flat)

    # Lift section, separately sampled so both boundaries are exact fit points.
    n_lift = max(7,p.fusion_lift_points)
    theta_lift = np.linspace(lift_start,extent,n_lift)

    lower_flat = interp_curve(
        theta_lower,
        lower_exact,
        theta_flat,
    )
    lower_lift = interp_curve(
        theta_lower,
        lower_exact,
        theta_lift,
    )

    # Phillips curve has an explicit normalized u parameter.
    u_exact = np.linspace(0.0,1.0,len(terminal_exact))
    u_fit = np.linspace(
        0.0,
        1.0,
        max(8,p.fusion_terminal_points),
    )
    terminal_fit = interp_curve(
        u_exact,
        terminal_exact,
        u_fit,
    )

    # Avoid duplicate fit points at section boundaries.
    fit = np.vstack((
        lower_flat,
        lower_lift[1:],
        terminal_fit[1:],
    ))

    if len(fit) > p.fusion_max_points:
        raise RuntimeError(
            f"Finished Fusion point count {len(fit)} exceeds "
            f"fusion_max_points={p.fusion_max_points}."
        )

    return fit


def build_unbent_fusion_points(
    p,
    g,
    unbent_exact,
    cut,
):
    n = max(
        8,
        int(math.ceil(
            cut["cut_turns"]*p.fusion_points_per_turn
        )) + 1,
    )

    theta_exact = np.linspace(
        0.0,
        cut["theta_cut"],
        len(unbent_exact),
    )
    theta_fit = np.linspace(
        0.0,
        cut["theta_cut"],
        n,
    )

    fit = interp_curve(
        theta_exact,
        unbent_exact,
        theta_fit,
    )

    if len(fit) > p.fusion_max_points:
        raise RuntimeError(
            f"Unbent Fusion point count {len(fit)} exceeds "
            f"fusion_max_points={p.fusion_max_points}."
        )

    return fit


def evaluate_fusion_fit(exact, fit, tolerance_mm):
    dense_count = max(30000, len(fit)*100)
    reconstructed = reconstruct_dense_from_fit(
        fit,
        count=dense_count,
    )

    err = geometric_spline_error(
        exact,
        reconstructed,
    )

    return reconstructed, dict(
        fit_point_count=len(fit),
        max_error_mm=err,
        max_error_um=1000.0*err,
        tolerance_um=1000.0*tolerance_mm,
        within_tolerance=err <= tolerance_mm,
    )


def export_csv(path,pts):
    np.savetxt(path,pts,delimiter=",",header="x_mm,y_mm,z_mm",comments="",fmt="%.9f")


def plot_manufacturing(path,p,g,pts,cut):
    fig,ax=plt.subplots(figsize=(8,8))
    ax.plot(pts[:,0],pts[:,1],label="Unbent Archimedean spring")

    xa,ya=xy_arch(np.array([cut["theta_active"]]),p,g)
    xc,yc=xy_arch(np.array([cut["theta_cut"]]),p,g)
    ax.scatter([xa[0]],[ya[0]],s=45,label="Active length")
    ax.scatter([xc[0]],[yc[0]],s=70,marker="x",label=f"Cut +{p.piton_extra_deg:.0f}°")

    ax.set_aspect("equal","box")
    ax.grid(True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Unbent manufacturing spring")
    ax.legend(fontsize=8)
    ax.text(0.02,0.02,
            f"active = {cut['active_turns']:.6f} turns\ncut = {cut['cut_turns']:.6f} turns",
            transform=ax.transAxes,va="bottom")
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_compare_2d(path,exact,rec,fit,title,rep):
    fig,ax=plt.subplots(figsize=(8,8))
    ax.plot(exact[:,0],exact[:,1],linewidth=1.5,label="Exact curve")
    ax.plot(rec[:,0],rec[:,1],"--",linewidth=1.0,label="Spline from reduced fit points")
    ax.scatter(fit[:,0],fit[:,1],s=8,label=f"Fit points ({len(fit)})")
    ax.set_aspect("equal","box")
    ax.grid(True)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.text(0.02,0.02,
            f"fit points = {rep['fit_point_count']}\nmax error = {rep['max_error_um']:.3f} µm",
            transform=ax.transAxes,va="bottom")
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_finished_3d(path,exact,rec,fit,g,rep):
    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection="3d")
    ax.plot(exact[:,0],exact[:,1],exact[:,2],linewidth=1.4,label="Exact 3D centerline")
    ax.plot(rec[:,0],rec[:,1],rec[:,2],"--",linewidth=1.0,label="Reduced spline")
    ax.scatter(fit[:,0],fit[:,1],fit[:,2],s=7,label=f"Fit points ({len(fit)})")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title(f"Finished spring: 60° lift to z={g['upper_z']:.3f} mm")
    ax.legend(fontsize=8)
    try: ax.set_box_aspect((1,1,0.35))
    except: pass
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_both_3d(path,unbent,finished,p,g):
    fig=plt.figure(figsize=(9,7))
    ax=fig.add_subplot(111,projection="3d")
    ax.plot(unbent[:,0],unbent[:,1],unbent[:,2],linewidth=1.1,label=f"Unbent z={p.unbent_z:.1f} mm")
    ax.plot(finished[:,0],finished[:,1],finished[:,2],linewidth=1.4,label="Finished Breguet spring")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title("Two 3D models for Fusion")
    ax.legend(fontsize=8)
    try: ax.set_box_aspect((1,1,0.55))
    except: pass
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    p=SpringParameters()
    g=derive(p)

    lower,theta_lower,lift_profile=lower_finished_exact(p,g)
    terminal,ph=optimise_terminal(p,g)
    finished=np.vstack((lower,terminal[1:]))

    cut=manufacturing_cut(p,g,finished)
    unbent=unbent_exact(p,g,cut)

    # Direct parametric fit points for Fusion.
    ffit=build_finished_fusion_points(
        p,g,lower,theta_lower,terminal
    )
    ufit=build_unbent_fusion_points(
        p,g,unbent,cut
    )

    frec,frep=evaluate_fusion_fit(
        finished,ffit,p.fusion_tolerance_mm
    )
    urec,urep=evaluate_fusion_fit(
        unbent,ufit,p.fusion_tolerance_mm
    )

    out=Path(__file__).resolve().parent

    export_csv(out/"finished_spring_exact.csv",finished)
    export_csv(out/"unbent_spring_exact.csv",unbent)
    export_csv(out/"finished_spring_fusion_fit_points.csv",ffit)
    export_csv(out/"unbent_spring_fusion_fit_points.csv",ufit)

    plot_manufacturing(
        out/"01_unbent_manufacturing_cut.png",
        p,g,unbent,cut
    )
    plot_compare_2d(
        out/"02_finished_exact_vs_reduced.png",
        finished,frec,ffit,
        "Finished spring — exact vs Fusion fit spline",
        frep
    )
    plot_finished_3d(
        out/"03_finished_3d.png",
        finished,frec,ffit,g,frep
    )
    plot_compare_2d(
        out/"04_unbent_exact_vs_reduced.png",
        unbent,urec,ufit,
        "Unbent spring — exact vs Fusion fit spline",
        urep
    )
    plot_both_3d(
        out/"05_both_models.png",
        unbent,finished,p,g
    )

    print("Phillips / Breguet spring")
    print("============================")
    print(f"Inner diameter                  : {p.inner_diameter:.6f} mm")
    print(f"Spring thickness                : {p.spring_thickness:.6f} mm")
    print(f"Spring height                   : {p.spring_height:.6f} mm")
    print(f"Nominal wound turns             : {p.turns:.6f}")
    print(f"Winding factor                  : {p.winding_factor:.6f}")
    print(f"Radial pitch / turn             : {g['radial_pitch']:.6f} mm")
    print(f"Diametral growth / turn         : {g['diametral_pitch']:.6f} mm")
    print(f"Nominal wound outer diameter    : {g['nominal_outer_diameter']:.6f} mm")
    print(f"Nominal wound outer radius      : {g['nominal_outer_radius']:.6f} mm")
    print(f"rho0 at terminal birth          : {g['rho0']:.6f} mm")
    print(f"Phillips number N               : {g['phillips_number']:.6f}")
    print(f"Attachment angle gamma          : {p.attachment_angle_deg:.3f} deg")
    print(f"Beta                            : {p.beta_deg:.3f} deg")
    print(f"Lower effective turns           : {g['lower_turns_effective']:.9f}")
    print()
    print("3D lift")
    print(f"Lift angle                      : {p.lift_angle_deg:.3f} deg")
    print(f"Lift height                     : {g['lift_height']:.6f} mm")
    print(f"Kick radius                     : {p.kick_radius:.6f} mm")
    print(f"Maximum lift inclination        : {lift_profile['alpha_deg']:.6f} deg")
    print(f"Middle tangent length           : {lift_profile['middle_length']:.6f} mm")
    print(f"Upper level                     : {g['upper_z']:.6f} mm")
    print(f"Unbent Fusion level             : {p.unbent_z:.6f} mm")
    print()
    print("Phillips conditions")
    print(f"m_x                             : {ph['mx']:.9f} mm^2")
    print(f"target rho0^2                   : {ph['target_mx']:.9f} mm^2")
    print(f"m_y                             : {ph['my']:.9f} mm^2")
    print(f"m_x error                       : {ph['mx_error_pct']:.6f} %")
    print(f"m_y error                       : {ph['my_error_pct']:.6f} %")
    print()
    print("Manufacturing")
    print(f"Finished active 3D length       : {cut['active_length']:.6f} mm")
    print(f"Unbent active turns             : {cut['active_turns']:.9f}")
    print(f"Cut turns (+5 deg)              : {cut['cut_turns']:.9f}")
    print(f"Cut angle                       : {cut['theta_cut_deg']:.6f} deg")
    print()
    print("Fusion spline reduction")
    print(f"Maximum allowed fit points      : {p.fusion_max_points}")
    print(f"Points / flat turn              : {p.fusion_points_per_turn}")
    print(f"Lift fit points                 : {p.fusion_lift_points}")
    print(f"Terminal fit points             : {p.fusion_terminal_points}")
    print(f"Finished exact points           : {len(finished)}")
    print(f"Finished Fusion fit points      : {frep['fit_point_count']}")
    print(f"Finished geometric spline error : {frep['max_error_um']:.4f} µm")
    print(f"Unbent exact points             : {len(unbent)}")
    print(f"Unbent Fusion fit points        : {urep['fit_point_count']}")
    print(f"Unbent geometric spline error   : {urep['max_error_um']:.4f} µm")
    print(f"Target geometric tolerance      : {p.fusion_tolerance_mm*1000:.3f} µm")
    print(f"Phillips optimizer success      : {ph['success']}")


if __name__=="__main__":
    main()

