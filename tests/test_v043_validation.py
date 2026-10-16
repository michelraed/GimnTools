"""Independent numerical & pipeline regressions for GimnToolsV2 0.4.3.

Not equivalent to scanner validation or NEMA compliance.
"""
import json
import numpy as np
import pytest
from Gimntools.GimnRec.reconstructor import siddon_reconstructor
from Gimntools.GimnRec.reconstruction_v2 import (apply_cor_shift,
                                                       apply_angular_offset,
                                                       process_file_v2)
from Gimntools.GimnRec.filters import interpolate_distance, smooth_angular, ramLak
from Gimntools.GimnRec.quality_metrics import measure_point_source
from Gimntools.GimnRec.systemSpace import SiPM
from Gimntools.GimnRec.sinogramer import convert_crystal_to_xy


def test_radon_180_parity_for_positive_and_negative_shift():
    sino = np.zeros((1, 9, 18))
    sino[0, 7, -1] = 100
    theta = (np.arange(18) + .5) * 10
    a = apply_angular_offset(sino, theta, 10)
    assert a[0, 1, 0] == 100
    assert a.sum() == pytest.approx(100)
    b = apply_angular_offset(a, theta, -10)
    np.testing.assert_allclose(b, sino, atol=1e-12)


@pytest.mark.parametrize('phi', [-47, -10, 10, 49, 180, 360])
def test_radon_offset_conserves_nonnegative_counts(phi):
    rng = np.random.default_rng(41)
    sino = rng.uniform(0, 10, (2, 17, 18))
    shifted = apply_angular_offset(sino, (np.arange(18)+.5)*10, phi)
    assert shifted.min() >= 0
    assert shifted.sum() == pytest.approx(sino.sum(), rel=1e-13)


def test_cor_interpolation_does_not_create_negative_counts():
    s = np.zeros((1, 17, 6)); s[0, 8, :] = 100
    result = apply_cor_shift(s, .45, -.14, np.arange(17)-8,
                             (np.arange(6)+.5)*30)
    assert result.min() >= 0
    assert np.all(np.isfinite(result))


def test_conservative_gui_path_not_cubic():
    s = np.zeros((1, 11, 3)); s[0, 5, :] = 100
    r = interpolate_distance(s, n_out=44, kind='conservative')
    assert r.min() >= 0
    assert r.sum() == pytest.approx(s.sum(), rel=1e-12)
    from pathlib import Path
    # Headless-compatible source audit: the GUI must not use cubic for counts.
    gui_source = (Path(__file__).resolve().parents[1] / 'src' / 'Gimntools' /
                  'GimnRec' / 'ui' / 'app.py').read_text()
    for name in ('class GatePanel', 'class PetsysPanel'):
        section = gui_source.split(name, 1)[1].split('class ', 1)[0]
        assert 'kind="cubic"' not in section


@pytest.mark.parametrize('side',[8, 16, 32])
def test_sipm_grid_and_crystal_mapping(side):
    sipm = SiPM(side, side, 1.75, 1.5)
    assert sipm.positionsX[0] == pytest.approx(-(side-1)/2 * 1.75)
    assert sipm.positionsX[-1] == pytest.approx((side-1)/2 * 1.75)
    for c in (0, side-1, side*(side-1), side**2-1):
        x,y = convert_crystal_to_xy(c, side, side)
        assert 0 <= x < side and 0 <= y < side
    with pytest.raises(ValueError):
        convert_crystal_to_xy(side**2, side, side)


def test_point_gaussian_axes_rotated():
    n = 101; fov=30.
    mm = (np.arange(n)+.5)*fov/n-fov/2
    X,Y = np.meshgrid(mm,mm)
    theta = np.deg2rad(45)
    u = X*np.cos(theta)+Y*np.sin(theta)
    v = -X*np.sin(theta)+Y*np.cos(theta)
    img = np.exp(-.5*(u*u+(v/2)**2))
    m = measure_point_source(img, fov, roi_mm=12)
    truth = 2.35482004503 / np.sqrt(.5 + .5 / 4)
    assert m['fwhm_x_mm'] == pytest.approx(truth, rel=.01)
    assert m['fwhm_y_mm'] == pytest.approx(truth, rel=.01)
    assert min(m['fwhm_principal_1_mm'],m['fwhm_principal_2_mm']) == pytest.approx(2.35482, rel=.01)


@pytest.mark.parametrize('input_sino', [np.array([[[np.nan]]]),
                                      np.array([[[np.inf]]]),
                                      np.zeros((0, 3, 2))])
def test_reconstructor_rejects_invalid_data(input_sino):
    with pytest.raises(ValueError):
        siddon_reconstructor(input_sino, fov_mm=10., verbose=False)


@pytest.mark.parametrize('bad_range', [0, -1, np.nan, np.inf])
def test_radial_range_rejected(bad_range):
    with pytest.raises(ValueError):
        siddon_reconstructor(np.ones((1, 8, 9)), fov_mm=10.,
                             sino_range_mm=bad_range, verbose=False)


def test_em_rejects_background_subtracted_count_histograms():
    sino = np.ones((1,8,8)); sino[0,0,0]=-1
    rec = siddon_reconstructor(sino, fov_mm=8., verbose=False)
    with pytest.raises(ValueError, match='background_sino'):
        rec.mlem((np.arange(8)+.5)*180/8, iterations=1, verbose=False)
    with pytest.raises(ValueError, match='background_sino'):
        rec.osem((np.arange(8)+.5)*180/8, iterations=1,
                 subsets_n=2, verbose=False)


@pytest.mark.parametrize('bad_bg', [np.ones((8,8))*-1,
                                   np.ones((8,8))*np.nan,
                                   np.ones((5,8))])
def test_em_rejects_invalid_background(bad_bg):
    rec = siddon_reconstructor(np.ones((1,8,8)), fov_mm=8., verbose=False)
    with pytest.raises(ValueError, match='background_sino'):
        rec.mlem((np.arange(8)+.5)*180/8, iterations=1,
                 verbose=False, background_sino=bad_bg)


def test_calibrated_projector_adjoint_for_angular_subsets():
    rec = siddon_reconstructor(np.ones((1, 13, 12)),
            fov_mm=22.0, n_img=18, sino_range_mm=19.0, verbose=False,
            rays_per_bin=3, backproject_chunks=2,
            cor_mm=.4, dx_cor_mm=.6, dy_cor_mm=-.3)
    rng = np.random.default_rng(99)
    img = rng.uniform(size=(18,18))
    angles = np.deg2rad((np.arange(12)+.5)*15 + 9.3)
    for th in [angles, angles[1::3]]:
        y = rng.uniform(size=(13,len(th)))
        lhs = np.vdot(rec._project(img, th),y)
        rhs = np.vdot(img, rec._backproject(y, th))
        assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_geometry_correction_matches_known_offsets():
    r = siddon_reconstructor(np.ones((1, 11, 5)), fov_mm=11., verbose=False,
                              cor_mm=.25, dx_cor_mm=1.0, dy_cor_mm=-2.0)
    a = np.deg2rad(np.array([0., 90.]))
    dist = r._distances_for_angles(a)
    np.testing.assert_allclose(dist[:,0],r.distances - 1)
    np.testing.assert_allclose(dist[:,1],r.distances + 2)


def _pll(y, mu):
    assert np.all(mu > 0)
    return float(np.sum(y*np.log(mu)-mu))


def test_poisson_mlem_background_log_likelihood_monotone():
    n=12; na=10
    theta = (np.arange(na)+.5)*180/na
    theta_rad=np.deg2rad(theta)
    seed=np.ones((1,n,na))
    r = siddon_reconstructor(seed, fov_mm=12., verbose=False, rays_per_bin=1)
    truth=np.zeros((n,n)); truth[3:9,3:9]=2.
    prompt = r._project(truth,theta_rad)+3.5
    r = siddon_reconstructor(prompt[None], fov_mm=12., verbose=False,
                              rays_per_bin=1)
    prev=_pll(prompt,r._project(np.ones((n,n)),theta_rad)+3.5)
    for it in range(1,6):
        x=r.mlem(theta,iterations=it,regularization_every=0, verbose=False,
                 ratio_cap=None, background_sino=np.full((n,na),3.5))[0]
        ll=_pll(prompt,r._project(x,theta_rad)+3.5)
        assert ll >= prev-1e-9
        assert np.all(np.isfinite(x)) and x.min()>=0
        prev=ll


def test_osem_single_subset_equals_mlem_with_background():
    theta=(np.arange(8)+.5)*180/8
    r0=siddon_reconstructor(np.zeros((1,9,8)),fov_mm=9.,verbose=False)
    phantom=np.zeros((9,9));phantom[3:6,2:6]=4
    mu=r0._project(phantom, np.deg2rad(theta))+1
    r=siddon_reconstructor(mu[None],fov_mm=9.,verbose=False)
    a=r.mlem(theta,iterations=3,regularization_every=0,verbose=False,
             ratio_cap=None,background_sino=np.ones((9,8)))
    b=r.osem(theta,iterations=3,subsets_n=1,regularization_every=0,
             verbose=False,ratio_cap=None,background_sino=np.ones((9,8)))
    np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)


def test_pipeline_background_method_selection_and_secure_npz(tmp_path):
    sino = np.zeros((1, 9, 8)); sino[0,4,:]=20
    bg = np.ones_like(sino)*2
    source_calls=[]
    def fetch(path,*args,**kwargs):
        source_calls.append(path)
        return (sino if path=='src.root' else bg).copy(), None
    kwargs=dict(fname='src.root',bgname='bg.root',
         cfg={'Half_FOV':6.,'crystal_size':[10.,12.]},
         det_angles={},params=None,rsec={},calibration={'dx_cor_mm':.12,
             'dy_cor_mm':-.18,'phi_offset_deg':2.5},
         coincidence_to_lor_fn=fetch,interpolate_distance_fn=interpolate_distance,
         smooth_angular_fn=smooth_angular,
         siddon_reconstructor_cls=siddon_reconstructor,ramLak_fn=ramLak,
         background_scale=2.0,methods=('mlem',),mlem_iters=2,
         gauss_every=0,angular_sigma=.5,save_intermediates=False,
         output_prefix=str(tmp_path)+'/')
    res=process_file_v2(**kwargs)
    assert source_calls==['src.root','bg.root']
    assert set(['mlem','sino_background','sino_raw','geometry']) <= res.keys()
    assert 'osem' not in res and 'fbp' not in res
    np.testing.assert_allclose(res['sino_background'],bg*2)
    np.testing.assert_allclose(res['sino_raw'],sino)
    assert res['geometry']['dx_cor_mm']==pytest.approx(.12)
    with np.load(tmp_path/'recon_src.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(f['sino_background'],bg*2)
        assert json.loads(str(f['geometry_json']))['dy_cor_mm']==pytest.approx(-.18)
        assert 'osem' not in f.files
        assert 'mlem' in f.files


def test_pipeline_rejects_invalid_exposure(tmp_path):
    dummy=lambda path,*args,**kwargs:(np.ones((1,8,8)),None)
    with pytest.raises(ValueError,match='background_scale'):
        process_file_v2('a','b',{'Half_FOV':5.,'crystal_size':[5.,5.]},
          {},{}, {}, {}, dummy, interpolate_distance, smooth_angular,
          siddon_reconstructor,ramLak, background_scale=-2,
          output_prefix=str(tmp_path)+'/')


@pytest.mark.parametrize('field,bad', [
    ('radius', 0), ('Half_FOV', -1), ('sino_range_mm', float('nan')),
    ('rotation_angles', 0), ('sipm_pixel_n', [8, 0]),
    ('crystal_size', [20., -1., 10.]), ('sipm_pitch',[1.,0.]),
    ('arround',float('inf')),
])
def test_strict_detector_config_rejects_invalid_physics(field,bad):
    from Gimntools.GimnRec.data_loaders import (load_detector_config,
                                                 require_valid_detector_config)
    from pathlib import Path
    path = Path(__file__).parents[1] / 'examples' / 'CONFIG_GATE' / 'detector_config_gate.json'
    cfg=load_detector_config(path)
    require_valid_detector_config(cfg, 'GATE')
    cfg[field]=bad
    with pytest.raises(ValueError):
        require_valid_detector_config(cfg, 'GATE')


def test_strict_petsys_energy_window():
    from Gimntools.GimnRec.data_loaders import load_detector_config, require_valid_detector_config
    from pathlib import Path
    cfg=load_detector_config(Path(__file__).parents[1] / 'examples' / 'CONFIG_PETSYS' / 'detector_config_petsys.json')
    require_valid_detector_config(cfg,'PETSYS')
    cfg['e_min'] = 651.
    with pytest.raises(ValueError,match='e_min < e_max'):
        require_valid_detector_config(cfg,'PETSYS')


def test_systemspace_rectangular_sipm_and_instance_state():
    from Gimntools.GimnRec.systemSpace import systemSpace
    from Gimntools.GimnRec.data_loaders import load_detector_config
    from pathlib import Path
    cfg=load_detector_config(Path(__file__).parents[1]/'examples'/'CONFIG_GATE'/'detector_config_gate.json')
    cfg['sipm_pixel_n']=np.array([8,12]);cfg['sipm_pitch']=np.array([3.0,2.0])
    a=systemSpace({'dA':0.},'y',cfg,'y')
    b=systemSpace({'dB':180.},'y',cfg,'y')
    assert len(a.positionsX)==8 and len(a.positionsY)==12
    assert a.positionsX[0]==pytest.approx(-10.5)
    assert a.positionsY[0]==pytest.approx(11)
    a.rotateGantry(30)
    assert 'dA' not in b.detectors and 'dB' not in a.detectors
    assert b.gantry_rotation == 0


@pytest.mark.parametrize('methods', ['mlem', 'fbp', 'osem,fbp'])
def test_gate_cli_uses_raw_count_data_and_safe_npz(monkeypatch, tmp_path, methods):
    from Gimntools.GimnRec.cli.reconstruct_gate import main
    from Gimntools.GimnRec import utils
    from pathlib import Path
    source=Path(__file__).parents[1]/'examples'/'CONFIG_GATE'
    f=tmp_path/'fake.root';f.touch()
    sino=np.zeros((1,9,8));sino[0,4,:]=10.
    monkeypatch.setattr(utils,'coincidence_to_lor',
                        lambda *a,**kw:(sino.copy(),None))
    out=tmp_path/'output'
    args=['--measurement',str(f), '--detector-config',str(source/'detector_config_gate.json'),
          '--detectors-angles',str(source/'detectors_angles.json'),
          '--rsectors',str(source/'rsectors_gate.json'),
          '--methods',methods, '--mlem-iters','2','--osem-iters','2',
          '--osem-subsets','2','--gauss-every','0',
          '--n-img','12','--n-dist-out','36', '--interpolate',
          '--angular-sigma','0.8', '--no-plots','--save-format','none',
          '--output-dir',str(out)]
    assert main(args)==0
    with np.load(out/'recon_fake.npz',allow_pickle=False) as res:
        np.testing.assert_allclose(res['sino_final'],sino)
        assert json.loads(str(res['geometry_json']))['n_img']==12
        for m in ('osem','mlem','fbp'):
            assert (m in res.files) == (m in methods.split(','))
        if 'fbp' in methods:
            assert res['sino_fbp'].shape[1]==36
        else:
            np.testing.assert_allclose(res['sino_fbp'],sino)
        for m in methods.split(','):
            assert res[m].shape==(1,12,12)


def test_bad_reconstructor_sampling_rejected():
    s=np.ones((1,9,8))
    for rays in (0,-1,0.4,float('nan'),'invalid'):
        with pytest.raises(ValueError,match='rays_per_bin'):
            siddon_reconstructor(s,fov_mm=12.,rays_per_bin=rays,verbose=False)
    for chunks in (0,-2,1.5,float('inf')):
        with pytest.raises(ValueError,match='backproject_chunks'):
            siddon_reconstructor(s,fov_mm=12.,backproject_chunks=chunks,verbose=False)


def test_gate_nonconsecutive_run_ids_preserve_true_angular_step(monkeypatch):
    """Run IDs 0,3,1 imply gantry changes -3*step,+2*step (not -step,-step)."""
    import sys
    import types
    from pathlib import Path
    from Gimntools.GimnRec.utils import coincidence_to_lor
    from Gimntools.GimnRec.systemSpace import systemSpace
    from Gimntools.GimnRec.data_loaders import load_detector_config
    cfg=load_detector_config(Path(__file__).parents[1]/'examples'/'CONFIG_GATE'/'detector_config_gate.json')
    cfg['process_as']='CrystalID'
    n=3
    fake={k: np.zeros(n) for k in (
        'globalPosX1','globalPosY1','globalPosZ1',
        'globalPosX2','globalPosY2','globalPosZ2',
        'crystalID1','crystalID2','rsectorID1','rsectorID2','runID')}
    fake['crystalID1']=np.array([16,16,16]);fake['crystalID2']=np.array([24,24,24])
    fake['rsectorID2']=np.ones(n,dtype=int)
    fake['runID']=np.array([0,3,1])
    branch=types.SimpleNamespace(arrays=lambda *a,**kw:fake)
    fakefile={'Coincidences':branch}
    monkeypatch.setitem(sys.modules,'uproot',types.SimpleNamespace(open=lambda x:fakefile))
    orig=systemSpace.rotateGantry
    called=[]
    def spy(self,rotation):
        called.append(rotation)
        return orig(self,rotation)
    monkeypatch.setattr(systemSpace,'rotateGantry',spy)
    sino, df=coincidence_to_lor('fake.root',cfg,
             {'detector-1':0.,'detector-2':180.},
             list(fake),rsectors={0:'detector-1',1:'detector-2'})
    np.testing.assert_allclose(called,[-45.,30.])
    assert sino.shape[0]==15 and len(df)==n
