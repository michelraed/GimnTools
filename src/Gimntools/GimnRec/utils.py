"""Sinogram building from coincidence data (ROOT files) and volume export.

Main entry points:
  - :func:`coincidence_to_lor` — build a (slice, distance, angle) sinogram
    from a GATE or PETSYS ROOT file.
  - :func:`save_reconstructions` — save reconstructed volumes as MHD/NRRD
    (Portuguese alias kept: :func:`salvar_reconstrucoes`).
"""

from scipy.spatial.transform import Rotation as ROT
from Gimntools.GimnRec.systemSpace import systemSpace
from Gimntools.GimnRec.sinogramer import convert_crystal_to_xy
import numpy as np
import pandas as pd
import os
import warnings


def _resolve_axes(detector_config):
    """Derive the system axes from detector_config['rotate_along'].

    In tomography the rotation axis is PARALLEL to the slice axis; the
    projection happens in the plane perpendicular to it. global_to_sinogram
    already follows this convention:
        rotate_along='z' -> project onto (x, y), slice = z (index 2)
        rotate_along='y' -> project onto (x, z), slice = y (index 1)
        rotate_along='x' -> project onto (y, z), slice = x (index 0)

    Returns (rotate_along, slice_axis_index).

    METHODOLOGY NOTE (GimnToolsV2): in the original code the gantry rotation
    axis was hard-coded to 'y' (systemSpace.rotateGantry/rotateBy and the
    systemSpace construction inside coincidence_to_lor), while the PROJECTION
    axis varied per `process_as` branch ('y' for GateGlobalPositions, 'z' for
    GeneralGlobalPosition). That is internally inconsistent whenever
    rotate_along != 'y'. Everything now derives from rotate_along.
    For rotate_along='y' the result is IDENTICAL to the old code; it only
    changes when a different axis is configured.
    """
    ra = str(detector_config.get("rotate_along", "y")).lower()
    if ra not in ("x", "y", "z"):
        raise ValueError(f"invalid rotate_along: {ra!r} (expected 'x', 'y' or 'z')")
    slice_axis_index = {"x": 0, "y": 1, "z": 2}[ra]
    return ra, slice_axis_index


def coincidence_to_lor(file_name, detector_config, detectors_angles, parameters,
                       matrix=[128,128,128], rsectors={0:"detector-1",1:"detector-2"},
                       first_rotation=None, verbose=False):
    """Build the (slice, distance, angle) sinogram from a ROOT file.

    Parameters
    ----------
    first_rotation : str or None
        Axis used to place the detectors initially. If None, uses the same
        axis as rotate_along (the geometrically consistent choice: detector
        ring perpendicular to the rotation axis). It can be overridden when
        the physical setup places the detectors on one axis and rotates the
        gantry about another.
    """
    # Fail early on invalid hardware configuration rather than processing
    # thousands of events with corrupt geometric coefficients.
    from Gimntools.GimnRec.data_loaders import require_valid_detector_config
    require_valid_detector_config(detector_config, detector_config.get("data_from", ""))
    import uproot as up   # ROOT dependency only when reading actual ROOT files
    rotate_along, slice_axis = _resolve_axes(detector_config)
    first_rot = (rotate_along if first_rotation is None
                 else str(first_rotation).lower())
    if first_rot not in ("x", "y", "z"):
        raise ValueError("first_rotation must be 'x', 'y' or 'z'")
    if verbose:
        print(f"[coincidence_to_lor] rotate_along={rotate_along!r}, "
              f"first_rotation={first_rot!r}, slice_axis_index={slice_axis}")

    PETSYS = systemSpace(detectors_angles, rotate_along, detector_config, first_rot)

    if detector_config["data_from"] == "GATE":
        f = up.open(file_name)
        c = f['Coincidences'].arrays(parameters, library='np')
        ev_n = c['globalPosX1'].shape[0]
        if detector_config["process_as"] in ("GateGlobalPositions","PaulGlobalPosition","GeneralGlobalPosition"):
            PETSYS.set_matrix(matrix)
        # astep is only used by the process_as branches that rotate the gantry
        # per runID; GeneralGlobalPosition does not, so don't require detNum there.
        if detector_config["process_as"] != "GeneralGlobalPosition":
            astep = (360/detector_config['rotation_angles'])/detector_config['detNum']
        else:
            astep = 0.0
        prev=0; df={'distance':[],'angle':[],'slice':[]}; cum=0
        for e in range(ev_n):
            cID1=c['crystalID1'][e]; cID2=c['crystalID2'][e]
            gX1,gY1,gZ1=c["globalPosX1"][e],c["globalPosY1"][e],c["globalPosZ1"][e]
            gX2,gY2,gZ2=c["globalPosX2"][e],c["globalPosY2"][e],c["globalPosZ2"][e]
            rS1=c["rsectorID1"][e]; rS2=c["rsectorID2"][e]; rID=c["runID"][e]
            if detector_config["process_as"]=="CrystalID":
                if rID != prev:
                    # Missing/unsorted runIDs must not silently compress
                    # angular spacing. Relative rotation is based on runID.
                    delta = int(rID) - int(prev)
                    PETSYS.rotateGantry(-astep * delta)
                    cum += astep * delta
                    prev = rID
                X1,Y1=convert_crystal_to_xy(cID1, int(detector_config['sipm_pixel_n'][0]), int(detector_config['sipm_pixel_n'][1])); X2,Y2=convert_crystal_to_xy(cID2, int(detector_config['sipm_pixel_n'][0]), int(detector_config['sipm_pixel_n'][1]))
                e1=np.asarray([-PETSYS.positionsX[X1],-PETSYS.positionsY[Y1],0])
                e2=np.asarray([-PETSYS.positionsX[X2],-PETSYS.positionsY[Y2],0])
                p1=PETSYS.getGlobalPreRotation(e1,rsectors[rS1]); p2=PETSYS.getGlobalPreRotation(e2,rsectors[rS2])
                g1=PETSYS.getGlobalRotation(p1); g2=PETSYS.getGlobalRotation(p2)
                d,a=PETSYS.global_to_sinogram(g1,g2,rotate_along)
                # CrystalID: the slice comes from the crystal row index (Y1, Y2),
                # which is local to the detector and independent of the rotation axis.
                sz=int(PETSYS.get_slice_seg((Y1,Y2)))
                df['distance'].append(d); df['angle'].append(a); df['slice'].append(sz); PETSYS.fill_sino(d,a,sz)
            elif detector_config["process_as"] in ("GateGlobalPositions","PaulGlobalPosition","GeneralGlobalPosition"):
                if detector_config["process_as"] != "GeneralGlobalPosition":
                    if rID != prev:
                        # Use true runID displacement, even if some runs are absent.
                        delta = int(rID) - int(prev)
                        PETSYS.rotateGantry(-astep * delta)
                        cum += astep * delta
                        prev = rID
                e1=np.asarray([gX1,gY1,gZ1]); e2=np.asarray([gX2,gY2,gZ2])
                g1=PETSYS.getGlobalRotation(e1); g2=PETSYS.getGlobalRotation(e2)
                d,a=PETSYS.global_to_sinogram(g1,g2,rotate_along)
                sz=int(PETSYS.get_slice_mono((g1[slice_axis],g2[slice_axis])))
                df['distance'].append(d); df['angle'].append(a); df['slice'].append(sz); PETSYS.fill_sino(d,a,sz)
        return PETSYS.get_sinogram(), pd.DataFrame(df)

    elif detector_config["data_from"] == "PETSYS":
        e_min=detector_config.get("e_min",450.0); e_max=detector_config.get("e_max",650.0)
        Rz = ROT.from_euler('z', detector_config.get("det_z_rotation",90.0), degrees=True)
        # rotate_along was already resolved/normalized above (lowercase).
        f = up.open(file_name)
        co = f['tomographicCoincidences'].arrays(parameters, library='np')
        ev_n = co['SiPMPosX_1'].shape[0]
        print(f"PETSYS: {ev_n} events — {os.path.basename(file_name)}")
        df={'distance':[],'angle':[],'slice':[]}; prev_a=None
        for e in range(ev_n):
            en1,en2=co["energyCorrected_1"][e],co["energyCorrected_2"][e]
            if en1<e_min or en1>e_max or en2<e_min or en2>e_max: continue
            ga=co["angleGantry_1"][e]
            if prev_a is None or ga!=prev_a: PETSYS.rotateBy(-ga); prev_a=ga
            c1,c2=co["chipID_1"][e],co["chipID_2"][e]
            d1n=rsectors.get(c1); d2n=rsectors.get(c2)
            if d1n is None or d2n is None: continue
            ev1=Rz.apply(np.asarray([co["SiPMPosX_1"][e],co["SiPMPosY_1"][e],0.0]))
            ev2=Rz.apply(np.asarray([co["SiPMPosX_2"][e],co["SiPMPosY_2"][e],0.0]))
            p1=PETSYS.getGlobalPreRotation(ev1,d1n); p2=PETSYS.getGlobalPreRotation(ev2,d2n)
            g1=PETSYS.getGlobalRotation(p1); g2=PETSYS.getGlobalRotation(p2)
            dist,ang=PETSYS.global_to_sinogram(g1,g2,rotate_along)
            sz=int(co["slice_1"][e])
            df['distance'].append(dist); df['angle'].append(ang); df['slice'].append(sz)
            PETSYS.fill_sino(dist,ang,sz)
        sino=PETSYS.get_sinogram(); df=pd.DataFrame(df)
        print(f"  Sino: {sino.shape}, max={sino.max():.0f}, LORs={len(df)}")
        return sino, df
    else:
        print("Unrecognized data_from value."); return None, None


def resolve_slice_thickness(detector_config, n_slices):
    """Axial spacing (mm) between reconstructed slices.

    The slice index of a coincidence is the SUM of the two axial coordinates
    (``Sinogramer.get_slice_seg`` for segmented crystals,
    ``Sinogramer.get_slice_mono`` for monolithic ones). As in any
    direct/cross-plane PET rebinning, the mid-plane of the LOR is HALF that
    sum, so consecutive slices are separated by half a detector element, not
    a full one.

    Resolution order:

    1. ``detector_config['slice_thickness_mm']`` if present -- always wins,
       and is the recommended way to pin the value down.
    2. Segmented crystals reached through ``process_as='CrystalID'``:
       ``sipm_pitch[1] / 2``, since the index is the sum of two crystal ROW
       indices spaced by the pitch.
    3. Monolithic rebinning (the ``*GlobalPosition*`` branches): the axial
       field of view in mid-plane coordinates is ``crystal_size[1]`` spread
       over ``n_slices`` bins, hence ``crystal_size[1] / n_slices``.

    Otherwise (in particular PETSYS data, where the slice index comes
    pre-computed in the ``slice_1`` branch and its convention is a property of
    the acquisition software, not of this package) the function falls back to
    ``sipm_pitch[1] / 2`` and warns. Set ``slice_thickness_mm`` explicitly to
    silence the warning.

    .. note:: **Corrected in 0.4.0.** The exporter previously wrote
       ``sipm_pitch[1]`` as the axial spacing, i.e. a FULL detector pitch.
       For the shipped 8x8 / 3.36 mm geometry that is 3.36 mm where the
       segmented rebinning gives 1.68 mm -- a factor of 2 in the axial
       direction of every exported MHD/NRRD header. Volumes exported before
       0.4.0 are axially stretched by that factor; any axial distance
       measured on them must be halved (or the header corrected).

    Parameters
    ----------
    detector_config : dict
    n_slices : int
        Number of slices in the volume actually being exported.

    Returns
    -------
    float
        Axial spacing in mm.
    """
    explicit = detector_config.get("slice_thickness_mm")
    if explicit is not None:
        return float(explicit)

    pitch_axial = float(np.asarray(detector_config["sipm_pitch"],
                                   dtype=float)[1])
    process_as = str(detector_config.get("process_as", "")).strip()
    data_from = str(detector_config.get("data_from", "")).upper()

    if data_from == "GATE" and process_as == "CrystalID":
        return pitch_axial / 2.0
    if data_from == "GATE" and process_as.endswith("GlobalPosition") \
            or process_as == "GateGlobalPositions":
        axial_fov = float(np.asarray(detector_config["crystal_size"],
                                     dtype=float)[1])
        return axial_fov / max(int(n_slices), 1)

    warnings.warn(
        "[export] the axial slice spacing could not be derived from "
        f"detector_config (data_from={data_from!r}, process_as={process_as!r}); "
        f"falling back to sipm_pitch[1]/2 = {pitch_axial / 2.0:.4f} mm. Set "
        "detector_config['slice_thickness_mm'] to the value your acquisition "
        "actually uses -- the exported header is only as trustworthy as this "
        "number.", stacklevel=3)
    return pitch_axial / 2.0


def _build_sitk_image(img_array, detector_config, n_dist_out=None,
                      fov_mm=None, voxel_mm=None, slice_thickness_mm=None):
    """Convert a ``(Z, Y, X)`` volume into a centred SimpleITK image.

    Shared by the MHD and NRRD exporters.

    The in-plane spacing is derived from the ACTUAL in-plane size of the
    volume (``fov_mm / img_array.shape[-1]``), not from ``n_dist_out``: the
    two are independent since the image grid became an explicit
    reconstruction parameter. ``n_dist_out`` is kept for call compatibility
    and only used to warn about a mismatch. ``voxel_mm`` overrides the
    in-plane spacing; ``slice_thickness_mm`` overrides the axial one
    (otherwise :func:`resolve_slice_thickness` decides).
    """
    img_array = np.asarray(img_array).astype(np.float32)
    n_xy = int(img_array.shape[-1])
    if fov_mm is None:
        fov_mm = detector_config["Half_FOV"] * 2.0
    fov_mm = float(fov_mm)

    if voxel_mm is not None:
        spacing_xy = float(voxel_mm)
        fov_mm = spacing_xy * n_xy
    else:
        spacing_xy = fov_mm / n_xy

    if n_dist_out is not None and int(n_dist_out) != n_xy:
        print(f"[NOTE] volume is {n_xy} px in-plane but n_dist_out="
              f"{int(n_dist_out)}; spacing derived from the volume "
              f"({spacing_xy:.4f} mm/px).")

    if slice_thickness_mm is not None:
        spacing_z = float(slice_thickness_mm)
    else:
        spacing_z = resolve_slice_thickness(detector_config,
                                            img_array.shape[0])

    import SimpleITK as sitk  # optional until volume export
    sitk_img = sitk.GetImageFromArray(img_array)  # SimpleITK reads [Z, Y, X]
    sitk_img.SetSpacing([spacing_xy, spacing_xy, spacing_z])  # order [X, Y, Z]
    origin_xy = -(fov_mm / 2.0) + (spacing_xy / 2.0)
    origin_z = -((img_array.shape[0] * spacing_z) / 2.0) + (spacing_z / 2.0)
    sitk_img.SetOrigin([origin_xy, origin_xy, origin_z])
    return sitk_img, (spacing_xy, spacing_xy, spacing_z)


def save_reconstructions(res_dict, detector_config, n_dist_out=None, output_dir=None,
                         fmt="mhd", keys=('osem', 'mlem', 'fbp'),
                         fov_mm=None, voxel_mm=None, slice_thickness_mm=None,
                         chaves_recon=None):
    """Save the reconstructions in the dict as MHD or NRRD via SimpleITK.

    fmt: 'mhd' (MetaImage, .mhd+.raw) or 'nrrd'. Both use SimpleITK, which
         is already a project dependency; NRRD adds no new dependency.

    fov_mm / voxel_mm: optional overrides for the in-plane geometry. Pass
         ``voxel_mm=rec.voxel_mm`` (from the reconstructor) when the image
         grid was chosen independently of the sinogram, so the exported
         header carries the true voxel size.

    slice_thickness_mm: optional override for the axial spacing. When None,
         :func:`resolve_slice_thickness` derives it from detector_config.

    chaves_recon: deprecated Portuguese alias of ``keys``.

    The Portuguese alias :func:`salvar_reconstrucoes` points to this same
    function and is kept for backward compatibility.
    """
    if chaves_recon is not None:
        keys = chaves_recon
    if output_dir is None:
        raise ValueError("output_dir is required")
    fmt = fmt.lower().lstrip('.')
    if fmt not in ("mhd", "nrrd"):
        raise ValueError(f"Unsupported format: {fmt!r} (use 'mhd' or 'nrrd')")
    os.makedirs(output_dir, exist_ok=True)
    base_name = res_dict.get('name', 'unnamed_reconstruction')

    saved = []
    for key in keys:
        img_array = res_dict.get(key)
        if img_array is None:
            continue
        sitk_img, sp = _build_sitk_image(
            img_array, detector_config, n_dist_out, fov_mm=fov_mm,
            voxel_mm=voxel_mm, slice_thickness_mm=slice_thickness_mm)
        out_filename = os.path.join(output_dir, f"{base_name}_{key}.{fmt}")
        import SimpleITK as sitk
        sitk.WriteImage(sitk_img, out_filename)
        saved.append(out_filename)
        print(f"[{fmt.upper()} saved] {out_filename} | "
              f"Spacing: {sp[0]:.3f}x{sp[1]:.3f}x{sp[2]:.3f} mm")
    return saved


# Portuguese aliases kept for backward compatibility with existing scripts.
salvar_reconstrucoes = save_reconstructions


def salvar_reconstrucoes_mhd(res_dict, detector_config, n_dist_out, output_dir):
    """Backward compatibility: save as MHD (delegates to save_reconstructions)."""
    return save_reconstructions(res_dict, detector_config, n_dist_out,
                                output_dir, fmt="mhd")
