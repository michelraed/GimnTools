import os
import numpy as np
import argparse
import scipy.io as sio

# Para visualização e reconstrução (importar as libs e funções conforme seu ambiente)
import uproot as up
from GimnTools.ImaGIMN.gimnRec.reconstructors.line_integral_reconstructor import line_integral_reconstructor
from GimnTools.ImaGIMN.gimnRec.reconstructors.system_matrix_reconstruction import reconstructor_system_matrix_cpu
from GimnTools.ImaGIMN.gimnRec.reconstructors.rotation_reconstruction import rotation_reconstructor
from GimnTools.ImaGIMN.processing.interpolators.reconstruction import bilinear_interpolation
from GimnTools.ImaGIMN.gimnRec.reconstruction_filters import ramLak


def GenerateSinogramFromPetsys(aqdName, BgName, e1_min, e1_max, e2_min, e2_max):
    file = up.open(aqdName)
    tomographicData = file["tomographicCoincidences"]
    data = tomographicData.arrays(["SiPMPosX_1","SiPMPosY_1","SiPMPosX_2","SiPMPosY_2",
                                  "rSino_1","angleSino_1","energyCorrected_1","energyCorrected_2",
                                  "slice_1","chipID_1","chipID_2","rSinoAnger_1","angleSinoAnger_1"], library="pd")

    pixel_n = 8
    rotation_angles = 12
    n_angles = int(((2 * pixel_n) - 1) * rotation_angles)
    n_distances = int((2 * pixel_n) - 1)
    n_slices = int((2 * pixel_n) - 1)
    matrix = np.asarray([n_slices, n_distances, n_angles])
    sinogram = np.zeros(matrix)

    # Leitura do background
    bgFile = up.open(BgName)
    tomographicDataBg = bgFile["tomographicCoincidences"]
    dataBg = tomographicDataBg.arrays(["SiPMPosX_1","SiPMPosY_1","SiPMPosX_2","SiPMPosY_2",
                                      "rSino_1","angleSino_1","energyCorrected_1","energyCorrected_2",
                                      "slice_1","chipID_1","chipID_2","rSinoAnger_1","angleSinoAnger_1"], library="pd")

    sinograms = []
    slices_unique = np.sort(data["slice_1"].unique())

    for slice_z in slices_unique:
        filtered = data[data["slice_1"] == slice_z]
        filtered = filtered[
            (filtered["energyCorrected_1"] >= e1_min) & (filtered["energyCorrected_1"] <= e1_max) &
            (filtered["energyCorrected_2"] >= e2_min) & (filtered["energyCorrected_2"] <= e2_max)
        ]

        bgfiltered = dataBg[dataBg["slice_1"] == slice_z]
        bgfiltered = bgfiltered[
            (bgfiltered["energyCorrected_1"] >= e1_min) & (bgfiltered["energyCorrected_1"] <= e1_max) &
            (bgfiltered["energyCorrected_2"] >= e2_min) & (bgfiltered["energyCorrected_2"] <= e2_max)
        ]

        hist_bg = np.histogram2d(bgfiltered["rSino_1"], bgfiltered["angleSino_1"], bins=[n_distances, n_angles])[0]
        hist_fg = np.histogram2d(filtered["rSino_1"], filtered["angleSino_1"], bins=[n_distances, n_angles])[0]
        result = hist_fg - hist_bg
        sinograms.append(result)

    return np.asarray(sinograms)


def reconstruct(path_acq, path_bg, method, algorithm, iterations, subsets,
                e1_min, e1_max, e2_min, e2_max):

    sino_3D = GenerateSinogramFromPetsys(path_acq, path_bg, e1_min, e1_max, e2_min, e2_max)
    angles = np.linspace(0, 180, sino_3D.shape[2], endpoint=True)

    if method == "LineIntegral":
        reconstructor = line_integral_reconstructor(sino_3D)
        if algorithm == "mlem":
            img = reconstructor.mlem(iterations * subsets, angles)
        elif algorithm == "osem":
            img = reconstructor.osem(iterations, subsets, angles)
        elif algorithm == "fbp":
            img = reconstructor.fbp(ramLak, angles)
        else:
            raise ValueError(f"Algoritmo desconhecido para LineIntegral: {algorithm}")

    elif method == "Rotation":
        reconstructor = rotation_reconstructor()
        reconstructor.set_sinogram(sino_3D)
        if algorithm == "mlem":
            img = reconstructor.mlem(iterations * subsets, bilinear_interpolation, angles)
        elif algorithm == "osem":
            img = reconstructor.osem(iterations, subsets, bilinear_interpolation, angles)
        elif algorithm == "fbp":
            img = reconstructor.fbp(bilinear_interpolation, ramLak, angles)
        else:
            raise ValueError(f"Algoritmo desconhecido para Rotation: {algorithm}")

    elif method == "SystemMatrix":
        stack = np.transpose(sino_3D, (0, 2, 1))
        reconstructor = reconstructor_system_matrix_cpu(stack)
        if algorithm == "mlem":
            img = reconstructor.mlem(iterations * subsets, angles)
        elif algorithm == "osem":
            img = reconstructor.osem(iterations, subsets, angles)
        elif algorithm == "fbp":
            img = np.zeros_like(reconstructor.mlem(iterations * subsets, angles))  # FBP não implementado aqui
        else:
            raise ValueError(f"Algoritmo desconhecido para SystemMatrix: {algorithm}")

    else:
        raise ValueError(f"Método de reconstrução desconhecido: {method}")

    return img


def save_results(img, output_prefix):
    # Salva numpy
    np.save(output_prefix + ".npy", img)
    # Salva .mat
    sio.savemat(output_prefix + ".mat", {"reconstructed_image": img})
    print(f"Salvo arquivo: {output_prefix}.npy e {output_prefix}.mat")


def main():
    parser = argparse.ArgumentParser(description="Reconstrução PET com parâmetros configuráveis.")
    parser.add_argument("--acquisition", "-a", required=True,
                        help="Caminho do arquivo ROOT da aquisição PET")
    parser.add_argument("--background", "-b", required=True,
                        help="Caminho do arquivo ROOT do background")
    parser.add_argument("--method", "-m", required=True, choices=["LineIntegral", "Rotation", "SystemMatrix"],
                        help="Método de reconstrução")
    parser.add_argument("--algorithm", "-alg", required=True, choices=["mlem", "osem", "fbp"],
                        help="Algoritmo de reconstrução")
    parser.add_argument("--iterations", "-i", type=int, default=2,
                        help="Número de iterações")
    parser.add_argument("--subsets", "-s", type=int, default=3,
                        help="Número de subsets (para OSEM)")
    parser.add_argument("--emin", type=float, default=490,
                        help="Energia mínima para filtragem")
    parser.add_argument("--emax", type=float, default=530,
                        help="Energia máxima para filtragem")
    parser.add_argument("--emin2", type=float, default=None,
                        help="Energia mínima para segunda energia (se não fornecido usa emin)")
    parser.add_argument("--emax2", type=float, default=None,
                        help="Energia máxima para segunda energia (se não fornecido usa emax)")
    parser.add_argument("--output", "-o", required=True,
                        help="Prefixo para salvar arquivos de saída (sem extensão)")

    args = parser.parse_args()

    e1_min = args.emin
    e1_max = args.emax
    e2_min = args.emin2 if args.emin2 is not None else e1_min
    e2_max = args.emax2 if args.emax2 is not None else e1_max

    print(f"Carregando aquisição: {args.acquisition}")
    print(f"Usando background: {args.background}")
    print(f"Método: {args.method}, Algoritmo: {args.algorithm}")
    print(f"Itarações: {args.iterations}, Subsets: {args.subsets}")
    print(f"Filtro de energia: E1 [{e1_min}, {e1_max}], E2 [{e2_min}, {e2_max}]")

    img = reconstruct(args.acquisition, args.background, args.method, args.algorithm,
                      args.iterations, args.subsets, e1_min, e1_max, e2_min, e2_max)

    print("Reconstrução concluída, salvando arquivos...")
    save_results(img, args.output)


if __name__ == "__main__":
    main()
