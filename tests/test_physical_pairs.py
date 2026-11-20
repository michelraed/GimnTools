"""Probability preservation, geometry validation and replay without Monte Carlo."""

from dataclasses import replace

import numpy as np
import pytest

from Gimntools.GimnRec.listmode import reconstruct_listmode
from Gimntools.GimnRec.listmode_response import ListmodeAcquisition, prepare_listmode_response
from Gimntools.GimnRec.opengate_calibration import _readout
from Gimntools.GimnRec.physical_response import read_physical_pairs, write_physical_pairs


def _acquisition():
    config = dict(data_from="GATE", radius=32., pla_thickness=2., Half_FOV=32.,
                  crystal_type="segmented", crystal_size=[15., 3.4, 3.4],
                  sipm_pitch=[3.4, 3.4], sipm_pixel_n=[1, 1], rotation_angles=2,
                  arround=180., detNum=2, rotate_along="z", process_as="CrystalID")
    meta = {"source": "root coincidences", "format": "GATE", "hit_positions": "crystal centres",
            "detector_config": config, "detector_angles_deg": {"left": 0., "right": 180.},
            "detector_mapping": {0: "left", 1: "right"},
            "poses": [{"step": 9, "angle_deg": 90., "exposure": 1.},
                      {"step": 2, "angle_deg": 0., "exposure": 3.}]}
    rays = np.array([[0., -34., 0., 0., 34., 0.], [34., 0., 0., -34., 0., 0.]])
    return rays, ListmodeAcquisition(meta, np.array([9, 2]))


def test_independent_matrix_reuses_zero_bins_and_reconstructs_without_sampling(tmp_path, monkeypatch):
    rays, records = _acquisition()
    shape, spacing, centre = (1, 3, 3), (2.,)*3, (0.,)*3
    table = prepare_listmode_response(rays, shape, spacing, centre, records, describe=True)
    assert table["pose_degrees"] == [0., 90.]
    emitted = np.full((2, 9), 100)
    params = dict(emitted=emitted, pose=[0, 1, 1], voxel=[4, 4, 1],
                  first=[0, 0, 0], second=[1, 1, 1], count=[10, 20, 25])
    path = tmp_path / "response.npz"
    provenance = {"engine": "synthetic calibration", "version": "1", "readout": "test readout"}
    write_physical_pairs(path, table, provenance, **params)
    physical = read_physical_pairs(path, table)
    detection, miss, uncertainty = physical.efficiency()
    assert detection[0, 4] == pytest.approx(.1) and miss[0, 4] == pytest.approx(.9)
    assert uncertainty[0, 4] == pytest.approx(.03)
    assert physical.pose_matrix(0).nnz == 1
    assert physical.pose_matrix(1).nnz == 2
    with pytest.raises(ValueError, match="differs in image_grid"):
        prepare_listmode_response(rays, shape, (1.,)*3, centre, records, physical_lut=path)
    with pytest.raises(ValueError, match="differs in detector"):
        prepare_listmode_response(rays, shape, spacing, centre, records, physical_lut=path,
                                  energy_window=(450., 650.))
    with pytest.raises(ValueError, match="exceed simulated primaries"):
        write_physical_pairs(path, table, provenance, **dict(params, count=[101, 20, 25]))
    monkeypatch.setattr("Gimntools.GimnRec.listmode_response._sample_pairs",
                        lambda *_: pytest.fail("physical calibration must be independent of analytical sampling"))
    response = prepare_listmode_response(rays, shape, spacing, centre, records, physical_lut=path)
    np.testing.assert_allclose(response.sensitivity[[1, 4]], [.25*.25, .25*.2 + .75*.1])
    a = response.operators[0]
    x = np.arange(9, dtype=float)
    y = np.array([.42])
    np.testing.assert_allclose((a @ x) @ y, x @ (a.T @ y))
    volume, details = reconstruct_listmode(rays, shape, spacing, detector_model=records,
                                            physical_lut=path, iterations=3, relaxation=1.)
    assert details["sensitivity"] == "physical crystal-pair response"
    assert np.argmax(volume) == 4
    assert np.sum(volume.ravel() * response.sensitivity) == pytest.approx(2.)

    incomplete = tmp_path / "incomplete.npz"
    write_physical_pairs(incomplete, table, provenance, emitted=emitted,
                         pose=[0], voxel=[4], first=[0], second=[1], count=[10])
    with pytest.raises(ValueError, match="missing 1 measured crystal pairs at pose 90"):
        prepare_listmode_response(rays, shape, spacing, centre, records, physical_lut=incomplete)


def test_calibration_digitizer_applies_energy_and_keeps_crystal_identity():
    first, second = _readout(np.array([0, 0, 0, 1, 1, 2, 2]), np.array([0, 1, 3, 0, 2, 0, 2]),
                             np.array([170., 330., 490., 410., 510., 520., 700.]),
                             4, 2, (450., 650.))
    np.testing.assert_array_equal(first, [1])
    np.testing.assert_array_equal(second, [3])
