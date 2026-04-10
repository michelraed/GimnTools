from GimnTools.ImaGIMN.image import image
from GimnTools.ImaGIMN.gimnRec.backprojectors import *
from GimnTools.ImaGIMN.gimnRec.projectors import *
from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import *
from GimnTools.ImaGIMN.processing.interpolators.reconstruction import *
from GimnTools.ImaGIMN.gimnRec.corrections import *
import numpy as np



def _safe_div(num, den, rel_eps=1e-8):
    den = np.asarray(den, dtype=np.float32)
    scale = max(float(np.max(den)), 1.0)
    return num / np.maximum(den, rel_eps * scale)

def _fov_mask(N):
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    r2 = (yy - c)**2 + (xx - c)**2
    return (r2 <= (c**2)).astype(np.float32)

def _fov_mask(N, margin_px=0):
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    r = max(c - float(margin_px), 0.0)
    return ((yy - c)**2 + (xx - c)**2 <= r**2).astype(np.float32)

def _floor_sensitivity(s, mask=None, q=10.0, abs_floor=1e-8):
    if mask is None:
        vals = s.ravel()
    else:
        vals = s[mask > 0]
    if vals.size == 0:
        return np.maximum(s, abs_floor)
    floor = max(float(np.percentile(vals, q)), abs_floor)
    return np.maximum(s.astype(np.float32), floor)

# --- Worker “uma fatia” do OSEM ---
def _osem_opt_worker(args):
    """
    Executa OSEM numa ÚNICA fatia.
    args = (sinogram_z, angles_subset, sens_image, pixels, iterations, use_fov_mask, GAUSS_EVERY, GAUSS_SIGMA)
    """
    sinogram, angles_subset, sens_image, pixels, iterations, use_fov_mask, GAUSS_EVERY, GAUSS_SIGMA = args

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    # FOV opcional
    fov_mask = _fov_mask(pixels) if use_fov_mask else None

    reconstruction = np.ones((pixels, pixels), dtype=np.float32)
    if use_fov_mask:
        reconstruction *= fov_mask

    # Subsets da fatia, alinhados com angles_subset/sens_image
    subsets_n = len(angles_subset)
    subsets = np.array_split(sinogram.astype(np.float32), subsets_n, axis=1)

    for it in range(iterations):
        for i, subset in enumerate(subsets):
            # forward (sem abs) + não-negatividade por clip
            x_nonneg = np.clip(reconstruction, 0.0, None)
            rec_sub  = direct_radon(x_nonneg, angles_subset[i]).astype(np.float32)

            # razão segura
            coef = _safe_div(subset, np.maximum(rec_sub, 0.0))
            coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
            coef[coef > 100] = 1  

            # backproj (sem abs) e normalização por sens do subset
            backprojection = inverse_radon(coef, angles_subset[i]).astype(np.float32)
            update = _safe_div(backprojection, sens_image[i])
            update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

            reconstruction *= update
            if use_fov_mask:
                reconstruction *= fov_mask
            reconstruction = np.clip(reconstruction, 0.0, None)

            # Gauss periódico (mantido)
            if GAUSS_EVERY > 0 and ((it + 1) % GAUSS_EVERY == 0):
                if gaussian_filter is None:
                    raise RuntimeError("scipy.ndimage não disponível. Instale SciPy ou use GAUSS_EVERY=0.")
                reconstruction = gaussian_filter(reconstruction, sigma=GAUSS_SIGMA, mode='nearest')

    return reconstruction

# ---- Worker: MLEM em UMA fatia, com safeguards adaptativos ----
def _mlem_worker_adaptive(args):
    """
    args = (sinogram, angles, sens_image, pixels,
            iterations, use_fov_mask, fov_mask,
            ratio_max, gauss_every, gauss_sigma_px,
            rel_eps, small_grid, update_clip_small)
    """
    (sinogram, angles, sens_image, pixels,
     iterations, use_fov_mask, fov_mask,
     ratio_max, gauss_every, gauss_sigma_px,
     rel_eps, small_grid, update_clip_small) = args

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    y = sinogram.astype(np.float32)
    x = np.ones((pixels, pixels), dtype=np.float32)
    if use_fov_mask:
        x *= fov_mask

    for it in range(int(iterations)):
        # forward (sem abs); não-negatividade por clip
        Ax = direct_radon(np.clip(x, 0.0, None), angles).astype(np.float32)

        # y/(Ax) com proteção + cap
        diff = _safe_div(y, np.maximum(Ax, 0.0), rel_eps=rel_eps)
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        if ratio_max is not None:
            diff = np.minimum(diff, float(ratio_max))

        # backproj (sem abs) e atualização multiplicativa
        back = inverse_radon(diff, angles).astype(np.float32)
        update = _safe_div(back, sens_image, rel_eps=rel_eps)
        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

        # relaxação para grade pequena
        if small_grid and update_clip_small is not None:
            umin, umax = update_clip_small
            update = np.clip(update, float(umin), float(umax))

        x *= update
        if use_fov_mask:
            x *= fov_mask
        x = np.clip(x, 0.0, None)

        # Gauss periódico (opcional)
        if gauss_every and gauss_every > 0 and ((it + 1) % gauss_every == 0):
            if gaussian_filter is None:
                raise RuntimeError("scipy.ndimage indisponível para GAUSS_EVERY>0.")
            x = gaussian_filter(x, sigma=float(gauss_sigma_px), mode='nearest')
            if use_fov_mask:
                x *= fov_mask
            x = np.clip(x, 0.0, None)

    return x

class line_integral_reconstructor(image):
    """
    @brief Creates a Reconstructor class that will inherit the image class.

    The sinogram order inside our program is:
    - rows: slice
    - columns: distances
    - Z: angles

    The sinogram order list must have the following names, but the order can change:
    - ("slice", "distances", "angles")
    """

    __sinogram_order_recon = ("slice", "distances", "angles")  # Sinogram order inside our program
    __center_of_rotation = None  # Center of rotation
    __sinogram_order = None  # Sinogram Order, used for repositioning the dimensions following the order needed to reconstruct
    __sinogram = None
    __reconstructed_mlem = None
    __reconstructed_osem = None
    __reconstructed_fbp = None

    def __init__(self, sinogram, center_of_rotation=None):
        """
        @brief Constructs the reconstructor class, it will initiate the super class of reconstructor, that inherits an image class.

        The image class will be responsible for opening the dicom and retrieving its pixels as a numpy object.

        @param path Path to the image file
        @param sinogram Sinogram data
        @param sinogram_order Order of the sinogram dimensions, default is ("slice", "angles", "distances")
        @param center_of_rotation Center of rotation
        @param transpose Flag to transpose the sinogram
        """
        super(line_integral_reconstructor, self).__init__(image=sinogram)
        self.__sinogram = sinogram

    @property
    def sinogram(self):
        """
        @brief Returns the sinogram pixels as a numpy object
        @return Sinogram as a numpy array
        """
        return self.__sinogram

    def set_sinogram(self, sinogram):
        """
        @brief Sets the sinogram data.

        @param sinogram Sinogram data to set
        """
        self.__sinogram = sinogram

    def set_img(self, img):
        """
        @brief Sets the image.

        @param img Image data to set
        """
        self.__img = img

    def set_center_of_rotation(self, center_of_rotation):
        """
        @brief Sets the value for the center of rotation.

        @param center_of_rotation Center of rotation value
        """
        self.__center_of_rotation = center_of_rotation

    def osem_MT(self, iterations, subsets_n, angles, verbose=False,
                GAUSS_EVERY=0, GAUSS_SIGMA=0.5,
                n_jobs=15, backend="process"):
        """
         Se n_jobs>1, paraleliza cada fatia.
        - backend: "process" (default) ou "thread"
        """
        import numpy as np
        import concurrent.futures as cf

        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]), dtype=np.float32)

        # --- Sensibilidade por subset (uma vez, fora dos workers) ---
        sino_ones = np.ones_like(self.sinogram[0], dtype=np.float32)
        angles_subset = np.array_split(angles, subsets_n)
        sino_ones_subset = np.array_split(sino_ones, subsets_n, axis=1)

        # FOV opcional
        use_fov_mask = False
        fov_mask = _fov_mask(pixels) if use_fov_mask else None

        sens_list = []
        for ss, subset_ones in enumerate(sino_ones_subset):
            s = inverse_radon(subset_ones.astype(np.float32), angles_subset[ss]).astype(np.float32)  # sem abs
            if use_fov_mask:
                s = s * fov_mask
                s = _floor_sensitivity(s, mask=fov_mask, q=10.0, abs_floor=1e-8)
            else:
                s = _floor_sensitivity(s, mask=None, q=10.0, abs_floor=1e-8)
            sens_list.append(s)
        sens_image = np.asarray(sens_list, dtype=np.float32)  # (subsets, pixels, pixels)

        # --- Execução por fatia ---
        if n_jobs is None or n_jobs <= 1:
            # Caminho sequencial 
            for z in range(slices):
                sinogram = self.sinogram[z, :, :]
                rec[z] = _osem_opt_worker((sinogram, angles_subset, sens_image, pixels, iterations,
                                        use_fov_mask, GAUSS_EVERY, GAUSS_SIGMA))
                if verbose:
                    print(f"[slice {z:02d}] done")
        else:
            # Paralelo por fatia
            tasks = []
            for z in range(slices):
                sinogram = self.sinogram[z, :, :]
                tasks.append((sinogram, angles_subset, sens_image, pixels, iterations,
                            use_fov_mask, GAUSS_EVERY, GAUSS_SIGMA))

            if backend == "thread":
                Executor = cf.ThreadPoolExecutor
                ctx = None
            else:
                import multiprocessing as mp
                # Em Linux/Jupyter, 'fork' costuma funcionar melhor. Em Windows/macos use script/guard.
                Executor = cf.ProcessPoolExecutor
                ctx = mp.get_context("fork")

            with Executor(max_workers=int(n_jobs), mp_context=ctx) if ctx else Executor(max_workers=int(n_jobs)) as ex:
                for z, recon in enumerate(ex.map(_osem_opt_worker, tasks, chunksize=1)):
                    rec[z] = recon
                    if verbose:
                        print(f"[slice {z:02d}] done (parallel)")

        self.__reconstructed_osem = rec
        return rec

    def osem_opt2(self, iterations, subsets_n, angles, verbose=False, GAUSS_EVERY=5, GAUSS_SIGMA=0.5):
        """
        OSEM com correções numéricas:
        - sem abs() em A e A^T
        - divisão estável (eps relativo)
        - sensibilidade por subset com piso percentílico + FOV opcional
        - não-negatividade por clip e bloqueio de NaN/Inf
        """
        import numpy as np
        try:
            from scipy.ndimage import gaussian_filter
        except Exception:
            gaussian_filter = None

        # ---------- helpers mínimos ----------
        def _safe_div(num, den, rel_eps=1e-8):
            den = np.asarray(den, dtype=np.float32)
            scale = max(float(np.max(den)), 1.0)
            return num / np.maximum(den, rel_eps * scale)

        def _fov_mask(N):
            yy, xx = np.mgrid[0:N, 0:N]
            c = (N - 1) / 2.0
            r2 = (yy - c)**2 + (xx - c)**2
            return (r2 <= (c**2)).astype(np.float32)

        def _floor_sensitivity(s, mask=None, q=10.0, abs_floor=1e-8):
            if mask is None:
                vals = s.ravel()
            else:
                vals = s[mask > 0]
            if vals.size == 0:
                return np.maximum(s, abs_floor)
            floor = max(float(np.percentile(vals, q)), abs_floor)
            return np.maximum(s.astype(np.float32), floor)

        # -----------------------------------------------------------------

        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]), dtype=np.float32)
        slice_count = self.slice_n()  # mantido (mesmo se não usar adiante)

        sino_ones = np.ones_like(self.sinogram[0], dtype=np.float32)
        angles_subset = np.array_split(angles, subsets_n)
        sino_ones_subset = np.array_split(sino_ones, subsets_n, axis=1)

        # ---- sensibilidade por subset, sem abs(), com piso e FOV opcional ----
        use_fov_mask = False  # mantém seu default
        fov_mask = _fov_mask(pixels) if use_fov_mask else None

        sens_image = []
        for ss, subset_ones in enumerate(sino_ones_subset):
            s = inverse_radon(subset_ones.astype(np.float32), angles_subset[ss]).astype(np.float32)  # sem abs()
            if use_fov_mask:
                s = s * fov_mask
                s = _floor_sensitivity(s, mask=fov_mask, q=10.0, abs_floor=1e-8)
            else:
                s = _floor_sensitivity(s, mask=None, q=10.0, abs_floor=1e-8)
            sens_image.append(s)
        sens_image = np.asarray(sens_image, dtype=np.float32)  # shape: (subsets, pixels, pixels)

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z, :, :].astype(np.float32)
            subsets = np.array_split(sinogram, subsets_n, axis=1)
            reconstruction = np.ones((pixels, pixels), dtype=np.float32)
            if use_fov_mask:
                reconstruction *= fov_mask  # inicializa só dentro do FOV

            iteration_counter = 0
            for it in range(iterations):
                iteration_counter += 1
                for i, subset in enumerate(subsets):
                    # ---- forward nos ângulos do subset (sem abs) ----
                    x_nonneg = np.clip(reconstruction, 0.0, None)  # não-negatividade sem distorcer adjunto
                    rec_sub = direct_radon(x_nonneg, angles_subset[i]).astype(np.float32)

                    # ---- razão segura y/(Ax) ----
                    coef = _safe_div(subset.astype(np.float32), np.maximum(rec_sub, 0.0))
                    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

                    coef[coef > 100] = 1

                    # ---- backproj do coef (sem abs) e normalização por sens do subset ----
                    backprojection = inverse_radon(coef, angles_subset[i]).astype(np.float32)
                    update = _safe_div(backprojection, sens_image[i])
                    update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

                    # ---- passo multiplicativo + suporte e não-negatividade ----
                    reconstruction *= update
                    if use_fov_mask:
                        reconstruction *= fov_mask
                    reconstruction = np.clip(reconstruction, 0.0, None)

                    # ---- Gauss periódico ----
                    if GAUSS_EVERY > 0 and ((it + 1) % GAUSS_EVERY == 0):
                        if gaussian_filter is None:
                            raise RuntimeError("scipy.ndimage não disponível para suavização. Instale SciPy ou use GAUSS_EVERY=0.")
                        reconstruction = gaussian_filter(reconstruction, sigma=GAUSS_SIGMA, mode='nearest')

                if verbose and ((it + 1) % max(1, iterations // 10) == 0):
                    # métrica simples de monitoramento
                    Ax_all = direct_radon(np.clip(reconstruction, 0.0, None), angles).astype(np.float32)
                    Ax_all = np.maximum(Ax_all, 1e-8)
                    ll = float((sinogram * np.log(Ax_all) - Ax_all).sum())
                    print(f"[slice {slice_z:02d}] it {it+1}/{iterations}  LL≈ {ll:.3e}")

            rec[slice_z, :, :] = reconstruction

        self.__reconstructed_osem = rec
        return rec


    def osem_opt(self, iterations, subsets_n, angles, verbose=False, GAUSS_EVERY = 5 ,GAUSS_SIGMA = 0.5):
        """
        @brief Reconstructs the sinogram using the Ordered Subset Expectation Maximization (OSEM) algorithm for a given number of iterations.

        This reconstruction is done using the rotations of the "reconstructed image" in order to obtain the projections.

        @param iterations Number of iterations
        @param subsets_n Number of subsets
        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param angles Angles for reconstruction, should be a numpy array of angles in radians
        @param verbose Flag to print the iteration number during reconstruction

        @return Reconstructed image using the OSEM algorithm
        """

        from matplotlib import pyplot as plt

        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        slice_count = self.slice_n()

        sino_ones = np.ones_like(self.sinogram[0])
        angles_subset = np.array_split(angles, subsets_n)
        sino_ones_subset = np.array_split(sino_ones,subsets_n,axis=1)
        sens_image = np.asarray([inverse_radon(subset,angles_subset[ss]) for ss,subset in enumerate(sino_ones_subset)])
                                                               
        use_fov_mask = False 
        if use_fov_mask:
                yy, xx = np.mgrid[0:pixels, 0:pixels]
                cx = (pixels-1)/2.0
                cy = (pixels-1)/2.0
                R  = ((pixels-1)/2.0)
                fov_mask = ((xx-cx)**2 + (yy-cy)**2) <= (R*R)
                fov_mask = fov_mask.astype(np.float32)
        else:
                fov_mask = None
        for slice_z in range(slices):

            sinogram = self.sinogram[slice_z, :, :]
            subsets = np.array_split(sinogram, subsets_n, axis=1)            
            reconstruction = np.ones([pixels, pixels])
            iteration_counter = 0
            for it in range(iterations):
                iteration_counter +=1    
                for i, subset in enumerate(subsets):
                    #aqui ele vai pegar a imagem estimada, e projetar apenas nos angulos do subset desejado
                    rec_sub = np.abs(direct_radon(reconstruction, angles_subset[i]))
                    rec_sub[rec_sub==0]=1 # se o valor for zero tira ele
                    #calcula-se a razão do subset pelo subset estimado nesta iteração
                    coef = np.nan_to_num((subset / (rec_sub+1e-10)),nan=1)

                    # isso daqui ta pra quando um pixel cresce demais, ele volta pra um, pra nao estragar. 
                    coef[coef>100]=1
                    backprojection = np.abs(inverse_radon(coef, angles_subset[i]))/(sens_image[i]+1e-9)
                    
                    reconstruction *= backprojection
                    if use_fov_mask:
                        reconstruction = reconstruction*fov_mask
                    else:
                        reconstruction= reconstruction
                     # -------- Filtro Gaussiano a cada 5 iterações --------
                    if GAUSS_EVERY > 0 and ((it + 1) % GAUSS_EVERY == 0):
                         reconstruction = gaussian_filter(reconstruction, sigma=GAUSS_SIGMA, mode='nearest')

            rec[slice_z, :, :] = reconstruction

        self.__reconstructed_osem = rec
        return rec


    def osem(self, iterations, subsets_n, angles, verbose=False):
        """
        @brief Reconstructs the sinogram using the Ordered Subset Expectation Maximization (OSEM) algorithm for a given number of iterations.

        This reconstruction is done using the rotations of the "reconstructed image" in order to obtain the projections.

        @param iterations Number of iterations
        @param subsets_n Number of subsets
        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param angles Angles for reconstruction, should be a numpy array of angles in radians
        @param verbose Flag to print the iteration number during reconstruction

        @return Reconstructed image using the OSEM algorithm
        """

        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        slice_count = self.slice_n()

        sino_ones = np.ones_like(self.sinogram[0])
        angles_subset = np.array_split(angles, subsets_n)
        sino_ones_subset = np.array_split(sino_ones,subsets_n,axis=1)
        sens_image = np.asarray([inverse_radon(subset,angles_subset[ss]) for ss,subset in enumerate(sino_ones_subset)])
                                                               
        use_fov_mask = False 
        if use_fov_mask:
                yy, xx = np.mgrid[0:pixels, 0:pixels]
                cx = (pixels-1)/2.0
                cy = (pixels-1)/2.0
                R  = (pixels-1)/2.0
                fov_mask = ((xx-cx)**2 + (yy-cy)**2) <= (R*R)
                fov_mask = fov_mask.astype(np.float32)
        else:
                fov_mask = None
        for slice_z in range(slices):

            sinogram = self.sinogram[slice_z, :, :]
            subsets = np.array_split(sinogram, subsets_n, axis=1)            
            reconstruction = np.ones([pixels, pixels])
            for it in range(iterations):
                
                for i, subset in enumerate(subsets):
                    #aqui ele vai pegar a imagem estimada, e projetar apenas nos angulos do subset desejado
                    rec_sub = np.abs(direct_radon(reconstruction, angles_subset[i]))
                    rec_sub[rec_sub==0]=1 # se o valor for zero tira ele
                    #calcula-se a razão do subset pelo subset estimado nesta iteração
                    coef = np.nan_to_num((subset / (rec_sub+1e-10)),nan=1)

                    # isso daqui ta pra quando um pixel cresce demais, ele volta pra um, pra nao estragar. 
                    coef[coef>1000]=1
                    backprojection = np.abs(inverse_radon(coef, angles_subset[i]))/(sens_image[i]+1e-12)

                    reconstruction *= backprojection

            if use_fov_mask:
                reconstruction = reconstruction*fov_mask
            else:
                reconstruction= reconstruction
            #reconstruction = (reconstruction/reconstruction.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = reconstruction

        self.__reconstructed_osem = rec
        return rec

    def fbp(self, filter_type, angles):
        """

        @brief Reconstructs the sinogram using the Filtered Back-Projection (FBP) algorithm.

        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param filter_type Filter to be used in the FBP, can be: cossineFilter or ramLak
        @param angles Angles for reconstruction, should be a numpy array of angles in radians

        @return Reconstructed image using the FBP algorithm
        """
        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1

        slices = self.sinogram.shape[0]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        slice_count = self.slice_n()

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z, :, :]
            filtered = apply_filter_to_sinogram(filter_type, sinogram)
            img = np.abs(inverse_radon(filtered, angles=angles))
            #imagem_estimada = (img/img.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = img
        self.__reconstructed_fbp = rec
        return rec

    def slice_n(self):
        """
        @brief Counts the total number of counts for each slice.

        @return Array containing the total counts for each slice
        """
        slice_count = np.zeros(self.sinogram.shape[0])
        for slice_sino in range(self.sinogram.shape[0]):
            slice_count[slice_sino] = self.sinogram[slice_sino, :, :].sum()
        return slice_count

    def normalize(self, image):
        """
        @brief Normalize the sinogram.

        @param image Image to normalize

        @return Normalized image
        """
        norm = np.zeros(image.shape)
        slices_n = self.slice_n()
        image = image[:]
        # Additional normalization logic can be added here
        return norm
    


    def mlem_mod2(self, iterations, angles, verbose=False):
        """
        MLEM: x *= (A^T (y/(Ax))) / s
        Salvaguardas adaptativas para grades pequenas (evitar explosão de pixels).
        """
        import numpy as np
        try:
            from scipy.ndimage import gaussian_filter
        except Exception:
            gaussian_filter = None  # só usado se GAUSS_EVERY>0

        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1

        # ---------------- Salvaguardas (parâmetros) ----------------
        REL_EPS      = 1e-8      # epsilon relativo nas divisões
        S_PCT_FLOOR  = 10.0      # percentil para piso da sensibilidade
        S_ABS_FLOOR  = 1e-8      # piso absoluto de s
        USE_FOV_MASK = False      # limitar atualizações ao FOV (recomendado)

        # estes serão ajustados dinamicamente conforme o tamanho da imagem:
        RATIO_MAX_LARGE   = 100.0
        RATIO_MAX_SMALL   = 50.0
        GAUSS_EVERY_LARGE = 0      # desliga por padrão em grade grande
        GAUSS_EVERY_SMALL = 0      # filtra com mais frequência em grade pequena
        GAUSS_SIGMA_PX    = 0.6    # sigma em pixels
        UPDATE_CLIP_SMALL = (0.2, 5.0)  # limita multiplicador quando grade é pequena
        # -----------------------------------------------------------

        def _safe_div(num, den, rel_eps=REL_EPS):
            den = np.asarray(den, dtype=np.float32)
            scale = max(float(np.max(den)), 1.0)
            return num / np.maximum(den, rel_eps * scale)

        def _fov_mask(N, margin_px=0):
            yy, xx = np.mgrid[0:N, 0:N]
            c = (N - 1) / 2.0
            r = c - float(margin_px)
            r = max(r, 0.0)
            return ((yy - c)**2 + (xx - c)**2 <= r**2).astype(np.float32)

        def _floor_sensitivity(s, mask=None, q=S_PCT_FLOOR, abs_floor=S_ABS_FLOOR):
            vals = s[mask > 0] if mask is not None else s.ravel()
            if vals.size == 0:
                return np.maximum(s, abs_floor)
            floor = max(float(np.percentile(vals, q)), abs_floor)
            return np.maximum(s.astype(np.float32), floor)

        slice_count = self.slice_n()  # mantido (mesmo se não usar)
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]  
        rec = np.ones((slices, pixels, pixels), dtype=np.float32)

        # Heurística: grade pequena?
        small_grid = (pixels <= 32)

        # Parâmetros adaptados
        RATIO_MAX   = RATIO_MAX_SMALL if small_grid else RATIO_MAX_LARGE
        GAUSS_EVERY = GAUSS_EVERY_SMALL if small_grid else GAUSS_EVERY_LARGE
        fov_mask    = _fov_mask(pixels, margin_px=(1 if small_grid and USE_FOV_MASK else 0)) if USE_FOV_MASK else None

        # === Sensibilidade s = A^T 1 (uma vez) ===
        sino_ones = np.ones_like(self.sinogram[0], dtype=np.float32)
        sens_image = inverse_radon(sino_ones, angles).astype(np.float32)  # sem abs
        if USE_FOV_MASK:
            sens_image = _floor_sensitivity(sens_image * fov_mask, mask=fov_mask)
        else:
            sens_image = _floor_sensitivity(sens_image, mask=None)

        # (opcional) suavizar levemente s em grade pequena para tirar zeros pontuais
        if small_grid and gaussian_filter is not None:
            sens_image = gaussian_filter(sens_image, sigma=0.5, mode='nearest')
            if USE_FOV_MASK:
                sens_image = _floor_sensitivity(sens_image * fov_mask, mask=fov_mask)
            else:
                sens_image = _floor_sensitivity(sens_image, mask=None)

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z].astype(np.float32)
            imagem_estimada = np.ones((pixels, pixels), dtype=np.float32)
            if USE_FOV_MASK:
                imagem_estimada *= fov_mask

            for it in range(iterations):
                # Forward (A x) — sem abs; não-negatividade via clip
                Ax = direct_radon(np.clip(imagem_estimada, 0.0, None), angles).astype(np.float32)

                # Razão y/(Ax) com proteção + cap
                diff = _safe_div(sinogram, np.maximum(Ax, 0.0))
                diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                if RATIO_MAX is not None:
                    diff = np.minimum(diff, float(RATIO_MAX))

                # Backprojection (A^T diff) — sem abs
                back = inverse_radon(diff, angles).astype(np.float32)

                # Update multiplicativo: x *= back / s
                update = _safe_div(back, sens_image)
                update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

                # Em grade pequena, limite o passo multiplicativo (relaxação leve)
                if small_grid:
                    umin, umax = UPDATE_CLIP_SMALL
                    update = np.clip(update, umin, umax)

                imagem_estimada *= update

                # Suporte + não-negatividade
                if USE_FOV_MASK:
                    imagem_estimada *= fov_mask
                imagem_estimada = np.clip(imagem_estimada, 0.0, None)

                # (Opcional) Gauss periódico (mais frequente em grade pequena)
                if GAUSS_EVERY and GAUSS_EVERY > 0 and ((it + 1) % GAUSS_EVERY == 0):
                    if gaussian_filter is None:
                        raise RuntimeError("scipy.ndimage não disponível; desative GAUSS_EVERY ou instale SciPy.")
                    imagem_estimada = gaussian_filter(imagem_estimada, sigma=float(GAUSS_SIGMA_PX), mode='nearest')
                    if USE_FOV_MASK:
                        imagem_estimada *= fov_mask
                    imagem_estimada = np.clip(imagem_estimada, 0.0, None)

                if verbose and ((it + 1) % max(1, iterations // 10) == 0):
                    Axm = np.maximum(direct_radon(np.clip(imagem_estimada, 0.0, None), angles).astype(np.float32), 1e-8)
                    ll = float((sinogram * np.log(Axm) - Axm).sum())
                    print(f"[slice {slice_z:02d}] it {it+1}/{iterations}  LL≈ {ll:.3e}")

            # (opcional) normalização por contagem do slice:
            # imagem_estimada = (imagem_estimada / max(imagem_estimada.sum(), 1e-8)) * slice_count[slice_z]

            rec[slice_z, :, :] = imagem_estimada

        self.__reconstructed_mlem = rec
        return rec

    def mlem_mod(self, iterations, angles, verbose=False):
        import numpy as np

        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1

        slice_count = self.slice_n()  # (mantido)
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]

        rec = np.ones((slices, pixels, pixels), dtype=np.float32)

        # === Sensibilidade s = A^T 1 (uma vez, pois não depende de z) ===
        sino_ones = np.ones_like(self.sinogram[0], dtype=np.float32)
        sens_image = inverse_radon(sino_ones, angles).astype(np.float32)+1e-32  # [FIX] sem abs
        fov_mask = _fov_mask(pixels)                                           # [FIX] suporte consistente
        #sens_image *= fov_mask

        # [FIX] Piso robusto para s (evita divisão por valores muito pequenos)
        # usa percentil para não inflar em regiões locais
        s_vals = sens_image[fov_mask > 0]
        s_floor = np.percentile(s_vals, 10.0) if s_vals.size else 1e-8
        s_floor = max(float(s_floor), 1e-8)
        sens_image = np.maximum(sens_image, s_floor)

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z].astype(np.float32)
            imagem_estimada = np.ones((pixels, pixels), dtype=np.float32)

            # [FIX] Inicialização somente dentro do FOV
            imagem_estimada *= fov_mask

            for it in range(iterations):
                # Forward projection (A x)
                # [FIX] garantir não-negatividade de x SEM usar abs (preserva adjunto)
                x_nonneg = np.clip(imagem_estimada, 0.0, None)
                proje_estimada = direct_radon(x_nonneg, angles).astype(np.float32)

                # [FIX] proteção estável no denominador (sem "setar 1")
                diff = _safe_div(sinogram, np.maximum(proje_estimada, 0.0))

                # Bloqueio de NaN/Inf em diff (sem inflar com 1)
                diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

                # Backprojection A^T (y / (Ax))
                backproj =inverse_radon(diff, angles).astype(np.float32)  # [FIX] sem abs

                # Atualização EM: x *= back / s
                update = backproj #/ sens_image
                update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)

                imagem_estimada *= update

                # [FIX] restringe ao suporte e mantém não-negatividade
                #imagem_estimada *= fov_mask
                imagem_estimada = np.clip(imagem_estimada, 0.0, None)

                if verbose and ((it + 1) % max(1, iterations // 10) == 0):
                    # métrica simples para inspeção
                    Ax = np.maximum(direct_radon(np.clip(imagem_estimada, 0, None), angles).astype(np.float32), 1e-8)
                    ll = float((sinogram * np.log(Ax) - Ax).sum())
                    print(f"[slice {slice_z:02d}] it {it+1}/{iterations}  LL≈ {ll:.3e}")

            # (opcional, se quiser normalizar pelos contadores do slice)
            # imagem_estimada = (imagem_estimada / max(imagem_estimada.sum(), 1e-8)) * slice_count[slice_z]

            rec[slice_z] = imagem_estimada

        self.__reconstructed_mlem = rec
        return rec
    
    def mlem_MT(self, iterations, angles, verbose=False,
         *, n_jobs=15, backend="thread"):
        """
        MLEM com safeguards adaptativos e paralelismo por fatia.
        - Mantém a lógica: x *= (A^T (y/(Ax))) / s
        - Paraleliza por fatia quando n_jobs>1 (thread ou process).
        """
        import numpy as np
        import concurrent.futures as cf

        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1

        # ======= parâmetros de salvaguarda  =======
        REL_EPS      = 1e-8
        S_PCT_FLOOR  = 10.0
        S_ABS_FLOOR  = 1e-8
        USE_FOV_MASK = True

        RATIO_MAX_LARGE   = 100.0
        RATIO_MAX_SMALL   = 50.0
        GAUSS_EVERY_LARGE = 0
        GAUSS_EVERY_SMALL = 7
        GAUSS_SIGMA_PX    = 0.5
        UPDATE_CLIP_SMALL = (0.2, 5.0)  # multiplicador mínimo/máximo (apenas grade pequena)
        # ====================================================================

        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]  
        rec = np.ones((slices, pixels, pixels), dtype=np.float32)

        small_grid = (pixels <= 32)
        ratio_max   = RATIO_MAX_SMALL if small_grid else RATIO_MAX_LARGE
        gauss_every = GAUSS_EVERY_SMALL if small_grid else GAUSS_EVERY_LARGE
        update_clip_small = UPDATE_CLIP_SMALL if small_grid else None
        fov_mask = _fov_mask(pixels, margin_px=(1 if (small_grid and USE_FOV_MASK) else 0)) if USE_FOV_MASK else None

        # === Sensibilidade s = A^T 1 (uma vez), com piso e FOV ===
        sino_ones = np.ones_like(self.sinogram[0], dtype=np.float32)
        sens_image = inverse_radon(sino_ones, angles).astype(np.float32)  # sem abs

        if USE_FOV_MASK:
            sens_image = _floor_sensitivity(sens_image * fov_mask, mask=fov_mask, q=S_PCT_FLOOR, abs_floor=S_ABS_FLOOR)
        else:
            sens_image = _floor_sensitivity(sens_image, mask=None, q=S_PCT_FLOOR, abs_floor=S_ABS_FLOOR)

        # (opcional) suavizar s levemente em grade pequena
        if small_grid:
            try:
                from scipy.ndimage import gaussian_filter
                sens_image = gaussian_filter(sens_image, sigma=0.5, mode='nearest')
                if USE_FOV_MASK:
                    sens_image = _floor_sensitivity(sens_image * fov_mask, mask=fov_mask, q=S_PCT_FLOOR, abs_floor=S_ABS_FLOOR)
                else:
                    sens_image = _floor_sensitivity(sens_image, mask=None, q=S_PCT_FLOOR, abs_floor=S_ABS_FLOOR)
            except Exception:
                pass

        # --- Execução por fatia (sequencial ou paralela) ---
        if n_jobs is None or n_jobs <= 1:
            for z in range(slices):
                rec[z] = _mlem_worker_adaptive((
                    self.sinogram[z].astype(np.float32), angles, sens_image, pixels,
                    iterations, USE_FOV_MASK, fov_mask,
                    ratio_max, gauss_every, GAUSS_SIGMA_PX,
                    REL_EPS, small_grid, update_clip_small
                ))
                if verbose:
                    print(f"[slice {z:02d}] done")
        else:
            tasks = [
                (self.sinogram[z].astype(np.float32), angles, sens_image, pixels,
                iterations, USE_FOV_MASK, fov_mask,
                ratio_max, gauss_every, GAUSS_SIGMA_PX,
                REL_EPS, small_grid, update_clip_small)
                for z in range(slices)
            ]

            if backend == "thread":
                Executor, ctx = cf.ThreadPoolExecutor, None
            else:
                import multiprocessing as mp
                Executor, ctx = cf.ProcessPoolExecutor, mp.get_context("fork")

            with Executor(max_workers=int(n_jobs), mp_context=ctx) if ctx else Executor(max_workers=int(n_jobs)) as ex:
                for z, xz in enumerate(ex.map(_mlem_worker_adaptive, tasks, chunksize=1)):
                    rec[z] = xz
                    if verbose:
                        print(f"[slice {z:02d}] done (parallel)")

        self.__reconstructed_mlem = rec
        return rec

    def mlem(self, iterations, angles, verbose=False):
        """
        @brief Reconstructs the sinogram using the Maximum Likelihood Expectation Maximization (MLEM) algorithm for a given number of iterations.

        This reconstruction is done using the rotations of the "reconstructed image" in order to obtain the projections.

        @param iterations Number of iterations
        @param interpolation Interpolator to be used, can be: linear_interpolation, beta_spline_interpolation, bilinear_interpolation, or beta_spline_interpolation_o5
        @param angles Angles for reconstruction, should be a numpy array of angles in radians
        @param verbose Flag to print the iteration number during reconstruction

        @return Reconstructed image using the MLEM algorithm
        """
        if self.sinogram is None:
            print("No sinogram Loaded")
            return -1
        
        slice_count = self.slice_n()
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.ones((self.sinogram.shape[0], self.sinogram.shape[1], self.sinogram.shape[1]))
        #Generate a sinogram of ones for sensitivity image calculation
        

        sino_ones = np.ones_like(self.sinogram[0])
        sens_image = inverse_radon(sino_ones,angles)
        # Reshape the sinogram into 1D for subset processing

        for slice_z in range(slices):
            sinogram = self.sinogram[slice_z]
            imagem_estimada = np.ones([pixels, pixels])
            for it in range(iterations):
                proje_estimada = np.abs(direct_radon(np.abs(imagem_estimada),angles))
                proje_estimada[proje_estimada==0]=1 
                diff = np.nan_to_num(sinogram / (proje_estimada+1e-10),nan=1)
                #diff[diff>1000] = 1
                #imagem_estimada *= np.abs(np.nan_to_num(inverse_radon(diff,angles),copy=True,nan=1))
                imagem_estimada *= np.abs(inverse_radon(diff,angles))/(sens_image+1e-9)
            
            #imagem_estimada = (imagem_estimada/imagem_estimada.sum())*slice_count[slice_z]
            rec[slice_z, :, :] = imagem_estimada
     
        self.__reconstructed_mlem = rec
        return rec

    def compute_tv_gradient(self, image, epsilon=1e-8):
        """
        Calcula o gradiente da TV usando diferenças para frente, evitando o efeito wrap-around.
        """
        grad = np.zeros_like(image)
        # Diferenças para a direção x (vertical)
        diff_x = np.zeros_like(image)
        diff_x[:-1, :] = image[1:, :] - image[:-1, :]
        # Diferenças para a direção y (horizontal)
        diff_y = np.zeros_like(image)
        diff_y[:, :-1] = image[:, 1:] - image[:, :-1]

        # Gradiente normalizado (suavização)
        grad_x = diff_x / (np.abs(diff_x) + epsilon)
        grad_y = diff_y / (np.abs(diff_y) + epsilon)

        # Divergência dos gradientes
        div_x = np.zeros_like(image)
        div_y = np.zeros_like(image)
        div_x[1:, :] = grad_x[1:, :] - grad_x[:-1, :]
        div_y[:, 1:] = grad_y[:, 1:] - grad_y[:, :-1]

        grad = div_x + div_y
        return grad

    def osem_tv(self, iterations, subsets_n, angles, beta=0.2, tv_epsilon=1e-8, verbose=False):
        """
        Reconstrução usando OSEM com regularização TV.
        
        Primeiro é aplicada a atualização OSEM e, em seguida, um passo de
        descida de gradiente é feito para reduzir o termo TV.
        """
        slices = self.sinogram.shape[0]
        pixels = self.sinogram.shape[1]
        rec = np.zeros((slices, pixels, pixels))
        slice_count = self.slice_n()

        # Cálculo da sensibilidade
        sino_ones = np.ones_like(self.sinogram[0])
        sens_image = inverse_radon(sino_ones, angles)
        sens_image = np.clip(sens_image, 1e-6, None)

        for slice_z in range(slices):
            current_counts = slice_count[slice_z]
            sinogram = self.sinogram[slice_z]
            # Inicialização: pode usar FBP para obter uma boa aproximação inicial
            reconstruction = inverse_radon(sinogram, angles)
            reconstruction = np.clip(reconstruction, 1e-6, None)
            # Dividir sinograma e ângulos em subconjuntos
            subsets = np.array_split(sinogram, subsets_n, axis=1)
            angle_subsets = np.array_split(angles, subsets_n)
            for it in range(iterations):
                # Atualização OSEM
                total_update = np.ones((pixels, pixels))
                for subset, angles_ss in zip(subsets, angle_subsets):
                    proj = direct_radon(reconstruction, angles_ss)
                    ratio = subset / (proj + 1e-10)
                    backproj = inverse_radon(ratio, angles_ss)
                    total_update *= backproj ** (1.0 / subsets_n)
                reconstruction *= total_update
                reconstruction = reconstruction / sens_image

                # Passo de regularização TV (descida de gradiente)
                grad_tv = self.compute_tv_gradient(reconstruction, tv_epsilon)
                reconstruction = reconstruction - beta * grad_tv
                reconstruction = np.clip(reconstruction, 1e-6, None)

                # Normalização para preservar o total de contagens da fatia
                factor = current_counts / (reconstruction.sum() + 1e-10)
                reconstruction *= factor

            rec[slice_z] = reconstruction
        return rec
    
