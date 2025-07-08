import os
import glob
import math
import json
import random
import itertools
import numpy as np
from pyxtal import pyxtal
from pymatgen.core import Lattice
from pymatgen.core import Structure
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction import xrd



FormToMin={"SiO2":"Quartz", "CaCO3":"Calcite", "FeS2":"Pyrite", "Fe3O4":"Magnetite"}
MinToID={"Quartz":0, "Calcite":1, "Pyrite":2, "Magnetite":3, "Other":4}
IDToMin={0:"Quartz", 1:"Calcite", 2:"Pyrite", 3:"Magnetite", 4:"Other"}


# Modelos no lineales para pares de minerales
models = {
    ('Quartz', 'Calcite'): lambda x: 6.974 * x**3.254 + 11.825 * x**19.277 + 0.133,
    ('Quartz', 'Pyrite'): lambda x: 3.910 * x**1.705 + 16.356 * x**16.475 + 0.442,
    ('Quartz', 'Magnetite'): lambda x: - 40.958 * x**12.133 + 61.048 * x**12.133 + 0.877,
    ('Calcite', 'Pyrite'): lambda x: 6.418 * x**2.976 + 7.650 * x**13.584 + 0.667,
    ('Calcite', 'Magnetite'): lambda x: 4.315 * x**4.719 + 7.411 * x**4.720 + 0.880,
    ('Pyrite', 'Magnetite'): lambda x: 0.426 * x**2 + 0.146 * x + 0.776,
}


def genPatt(mineral):

    MinAng = 4
    MaxAng = 70
    rang = MaxAng - MinAng
    stepsize = 0.02
    steps = int(rang / stepsize)
    max_shift = 0.2
    max_strain = 0.03
    max_texture = 0.2
    eta_0 = 0.6
    eta_1 = -0.01
    a1_a2_ratio=0.517

    theta = np.arange(MinAng, MaxAng, stepsize) / 2
    theta_rad = np.deg2rad(theta)
    lorentz_polarization = (1 + np.cos(2 * theta_rad) ** 2) / (np.sin(theta_rad) ** 2 * np.cos(theta_rad))
    LP_background=np.ones(steps)*0.05*lorentz_polarization
    gauss_background = np.random.normal(0, 0.9, steps)+LP_background

    # Parámetros U, V, W basados en configuraciones instrumentales típicas
    def get_caglioti_params(instrument="default"):
        if instrument == "high_resolution":  # Difractómetro de alta resolución
            return 0.012, -0.06, 0.0007
        elif instrument == "standard_lab":  # Difractómetro de laboratorio estándar
            return 0.027, -0.18, 0.0012
        elif instrument == "low_quality":  # Instrumento de baja precisión
            return 0.05, -0.26, 0.003
        else:
            return random.uniform(0, 0.08), random.uniform(-0.35, 0), random.uniform(0, 0.005)

    U, V, W = get_caglioti_params(np.random.choice(["default", "high_resolution", "standard_lab", "low_quality"]))


    if mineral.is_ordered:
        xtal_struc = pyxtal()  # Se crea una instancia de pyxtal
        xtal_struc.from_seed(mineral)  # Se inicializa pyxtal con la estructura del mineral
        strain_range = np.linspace(0.0, max_strain, 100)  # Se define un rango de deformaciones posibles
        current_strain = random.choice(strain_range)  # Se selecciona una deformación al azar dentro del rango
        xtal_struc.apply_perturbation(d_lat=current_strain, d_coor=0.0)  # Se aplica la deformación a la estructura
        mineral = xtal_struc.to_pymatgen()  # Se actualiza la estructura de referencia con la deformación aplicada

    else:
        diag_range = np.linspace(1 - max_strain, 1 + max_strain, 1000)  # Rango de deformaciones diagonales
        off_diag_range = np.linspace(0 - max_strain, 0 + max_strain, 1000)  # Rango de deformaciones fuera de la diagonal
        s11, s22, s33 = [random.choice(diag_range) for v in range(3)]
        s12, s13, s21, s23, s31, s32 = [random.choice(off_diag_range) for v in range(6)]
        sg = mineral.get_space_group_info()[1]  # Se obtiene el grupo espacial de la estructura

        # Clasificación del grupo espacial
        sg_class = ''
        if sg in list(range(195, 231)):
            sg_class = 'cubic'
        elif sg in list(range(16, 76)):
            sg_class = 'orthorhombic'
        elif sg in list(range(3, 16)):
            sg_class = 'monoclinic'
        elif sg in list(range(1, 3)):
            sg_class = 'triclinic'
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175)):
                sg_class = 'low-sym hexagonal/tetragonal'
            else:
                sg_class = 'high-sym hexagonal/tetragonal'

        # Definición de tensores de deformación según la clase del grupo espacial
        if sg_class in ['cubic', 'orthorhombic', 'monoclinic', 'high-sym hexagonal/tetragonal']:
            v1 = [s11, 0, 0]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v1 = [s11, s12, 0]
        elif sg_class == 'triclinic':
            v1 = [s11, s12, s13]

        if sg_class in ['cubic', 'high-sym hexagonal/tetragonal']:
            v2 = [0, s11, 0]
        elif sg_class == 'orthorhombic':
            v2 = [0, s22, 0]
        elif sg_class == 'monoclinic':
            v2 = [0, s22, s23]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v2 = [-s12, s22, 0]
        elif sg_class == 'triclinic':
            v2 = [s21, s22, s23]

        if sg_class == 'cubic':
            v3 = [0, 0, s11]
        elif sg_class == 'high-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'orthorhombic':
            v3 = [0, 0, s33]
        elif sg_class == 'monoclinic':
            v3 = [0, s23, s33]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'triclinic':
            v3 = [s31, s32, s33]

        strain_tensor = np.array([v1, v2, v3])  # Se crea el tensor de deformación
        strained_matrix = np.matmul(mineral.lattice.matrix, strain_tensor)  # Se aplica el tensor a la matriz del mineral
        strained_lattice = Lattice(strained_matrix)  # Se obtiene la red deformada
        mineral.lattice = strained_lattice  # Se actualiza la red de la estructura de referencia

    # Obtener patrones para CuKa1 y CuKa2
    patt_a1 = xrd.XRDCalculator(wavelength="CuKa1").get_pattern(mineral, two_theta_range=(MinAng, MaxAng))
    patt_a2 = xrd.XRDCalculator(wavelength="CuKa2").get_pattern(mineral, two_theta_range=(MinAng, MaxAng))

    # CuKa1
    bragg_ang_a1 = patt_a1.x
    intensity_a1 = patt_a1.y
    hkls_a1 = [v[0]['hkl'] for v in patt_a1.hkls]

    # CuKa2
    bragg_ang_a2 = patt_a2.x
    intensity_a2 = a1_a2_ratio * patt_a2.y  # Atenuar intensidades para CuKa2
    hkls_a2 = [v[0]['hkl'] for v in patt_a2.hkls]

    # Combinar datos en una estructura para un manejo más claro
    data = [
        {"bragg_angle": angle, "intensity": intensity, "hkl": hkl}
        for angle, intensity, hkl in zip(bragg_ang_a1, intensity_a1, hkls_a1)
    ] + [
        {"bragg_angle": angle, "intensity": intensity, "hkl": hkl}
        for angle, intensity, hkl in zip(bragg_ang_a2, intensity_a2, hkls_a2)
    ]
    bragg_ang0 = np.array([entry["bragg_angle"] for entry in data])
    intensity0 = np.array([entry["intensity"] for entry in data])
    #print(bragg_ang0)
    #print(intensity0)


    shift_range = np.linspace(-max_shift, max_shift, 1000)
    shift = random.choice(shift_range)
    bragg_ang1 = np.array(bragg_ang0) + shift  # Desplazamiento uniforme de la posición de los picos

    intensity1 = []

    # Diccionario con direcciones preferenciales típicas por sistema cristalino
    preferred_directions = {
        "cubic": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)],  # Orientaciones comunes en FCC/BCC
        "hexagonal": [(0, 0, 0, 1), (1, 0, 0, 0), (1, -1, 0, 0), (1, 0, -1, 0)],  # Planos más comunes en HCP
        "tetragonal": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # Similares a cúbicos pero anisotrópicos
        "orthorhombic": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # Similar a tetragonal
        "monoclinic": [(1, 0, 0), (0, 1, 0)],  # Dos direcciones predominantes
        "triclinic": [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Menos restricciones, pero aún hay planos preferidos
    }

    # Determinar el sistema cristalino de la fase actual
    sg = mineral.get_space_group_info()[1]

    if sg in range(195, 231):
        sg_class = "cubic"
    elif sg in range(16, 76):
        sg_class = "orthorhombic"
    elif sg in range(3, 16):
        sg_class = "monoclinic"
    elif sg in range(1, 3):
        sg_class = "triclinic"
    elif sg in range(76, 195):
        sg_class = "hexagonal" if sg in range(168, 175) else "tetragonal"

    # Seleccionar una dirección preferencial basada en el sistema cristalino
    prefDir = np.array(random.choice(preferred_directions[sg_class]))

    # Normalizar el vector preferencial para evitar problemas en el cálculo del ángulo
    prefDir = prefDir / np.linalg.norm(prefDir)

    # Definir el parámetro de orientación preferencial March-Dollase
    G = random.uniform(1 - max_texture, 1 + max_texture)  # Controla el grado de orientación preferida

    # Lista para almacenar intensidades modificadas
    intensity1 = []

    # Obtener la base recíproca de la red cristalina
    reciprocal_lattice = mineral.lattice.reciprocal_lattice

    for entry in data:
        peak = entry["intensity"]

        # Tomar el primer conjunto de índices de Miller (el más intenso)
        if mineral.lattice.is_hexagonal():
            # Para estructuras hexagonales, los índices son (h, k, i, l)
            h, k, i, l = entry["hkl"]  # Desempaquetar los índices de Miller-Bravais
        else:
            # Para estructuras no hexagonales, los índices son (h, k, l)
            h, k, l = entry["hkl"]  # Desempaquetar los índices de Miller

        # Calcular el vector normal al plano (hkl) en la base recíproca
        if mineral.lattice.is_hexagonal():
            # Para estructuras hexagonales, ignoramos el índice i
            n_hkl = h * reciprocal_lattice.matrix[0] + k * reciprocal_lattice.matrix[1] + l * reciprocal_lattice.matrix[2]
        else:
            # Para estructuras no hexagonales
            n_hkl = h * reciprocal_lattice.matrix[0] + k * reciprocal_lattice.matrix[1] + l * reciprocal_lattice.matrix[2]

        # Calcular el ángulo alpha entre el vector normal y la dirección preferida
        norm_n_hkl = np.linalg.norm(n_hkl)
        norm_pref = np.linalg.norm(prefDir)

        if norm_n_hkl == 0 or norm_pref == 0:
            alpha = 0  # Si el vector es nulo, se ignora el ajuste
        else:
            cos_alpha = np.dot(n_hkl, prefDir) / (norm_n_hkl * norm_pref)
            alpha = np.arccos(np.clip(cos_alpha, -1, 1))  # Asegurar que esté dentro de un rango válido

        # Aplicar la función de March-Dollase
        texture_factor = (G**2 * np.cos(alpha)**2 + np.sin(alpha)**2) ** (-1.5)

        # Ajustar la intensidad usando el factor de textura
        adjusted_intensity = peak * texture_factor
        intensity1.append(adjusted_intensity)


    # Crear array de picos con intensidades ajustadas
    patt1 = np.zeros((len(bragg_ang1), 2))
    patt1[:, 0] = bragg_ang1
    patt1[:, 1] = intensity1

    pattern1 = np.zeros(steps)

    H = np.zeros((patt1.shape[0], 1))
    H[:, 0] = np.sqrt(np.abs(U * (np.tan(patt1[:, 0] * (np.pi / 180) / 2)) ** 2 + V * np.tan(patt1[:, 0] * (np.pi / 180) / 2) + W))

    for x_val in range(steps):
        current_angle = MinAng + x_val * stepsize
        y_val = 0
        for xy_idx in range(patt1.shape[0]):
            angle = patt1[xy_idx, 0]
            inten = patt1[xy_idx, 1]

            if (current_angle - 5) < angle < (current_angle + 5):
                x_diff = current_angle - angle
                h = H[xy_idx, 0]
                eta=eta_0+eta_1*current_angle
                eta=max(0, min(eta, 1))
                const_g = 4 * np.log(2)
                gaus = (np.sqrt(const_g) / (np.sqrt(np.pi) * h)) * np.exp(-const_g * (x_diff / h) ** 2)
                lorentz = (2*h)/(np.pi*(h**2+4*x_diff**2))
                pseudo_voigt = eta * lorentz + (1 - eta) * gaus
                y_val += inten * pseudo_voigt

        pattern1[x_val] += y_val

    noisy_pattern1 = np.random.poisson(pattern1) + gauss_background

    # Normalizar
    max_value = max(noisy_pattern1)
    if max_value > 0:

        norm_pattern1=(100*noisy_pattern1)/max_value
    else:
        norm_pattern1 = np.zeros_like(noisy_pattern1)

    # Graficar el patrón final
    #plt.plot(np.linspace(MinAng, MaxAng, steps), norm_pattern1)
    #plt.xlabel("2Theta (degrees)")
    #plt.ylabel("Intensity")
    #plt.title("Synthetic XRD")
    #plt.show()

    return norm_pattern1


def savePatts(in_dir, patts_per_cif=1, save=False, save_path=""):

    files = glob.glob(in_dir.rstrip('/') + '/*.cif')  # Lista con la ruta de todos los CIFs en in_dir
    count = 0
    XRDs = []

    while count < patts_per_cif:  # Genera el número especificado de patrones por cada CIF
        for i, file in enumerate(files):
            try:
                mineral = Structure.from_file(file)  # Estructura generada a partir de cada CIF
            except Exception as e:
                print(e)
                print(file)
                continue

            parser = CifParser(file)
            data = parser.as_dict()
            for key in data.keys():
                if "_chemical_name_mineral" in data[key]:
                    mineral_name = data[key]["_chemical_name_mineral"]  # Busca el nombre del mineral en el CIF
                    break
                else:
                    formula = mineral.composition.reduced_formula  # Si no lo encuentra lo deduce de la fórmula química
                    try:
                        mineral_name = FormToMin[formula]
                    except:
                        mineral_name="Other"

            minQ = [0, 0, 0, 0, 0]
            try:
                minQ[MinToID[mineral_name]] = 100  # Genera la etiqueta del patrón
            except:
                minQ[MinToID["Other"]] = 100

            Pattern = genPatt(mineral)  # Genera el patrón
            XRDs.append((Pattern, minQ))

            if save:
                unique_name = f"{save_path}{mineral_name}_{i}_{count}.json"
                data_to_save = {'xrd': Pattern.tolist(), 'q': minQ}
                with open(unique_name, 'w') as json_file:
                    json.dump(data_to_save, json_file)

        count += 1

    return XRDs


def mixPatts(XRDs, n_fases=2, num_of_patts=1, save=False, save_path=""):
    miXRDs = []

    for _ in range(num_of_patts):
        # Selección de fases únicas
        selected_fases = []
        while len(selected_fases) < n_fases:
            candidate = random.choice(XRDs)
            if not any(np.array_equal(candidate[1], f[1]) for f in selected_fases):
                selected_fases.append(candidate)

        # Generar proporciones aleatorias
        proporciones = np.random.random(n_fases)
        proporciones /= proporciones.sum()
        minerals = [IDToMin[np.argmax(phase[1])] for phase in selected_fases]
        props = proporciones.tolist()

        # Calcular la intensidad integrada de cada fase
        base_intensities = [np.trapz(phase[0]) for phase in selected_fases]

        # Función de pérdida para la optimización
        def loss(s):
            error = 0
            for i, j in itertools.combinations(range(n_fases), 2):
                min_i, min_j = minerals[i], minerals[j]
                x = props[i] / (props[i] + props[j])

                # Obtener Y_ij según el modelo
                if (min_i, min_j) in models:
                    y_pred = models[(min_i, min_j)](x)
                elif (min_j, min_i) in models:
                    y_pred = 1 / models[(min_j, min_i)](1 - x)
                else:
                    y_pred = x / (1 - x)  # Lineal

                # Calcular la relación de intensidades actual
                I_ratio = (s[i] * base_intensities[i]) / (s[j] * base_intensities[j])

                # Acumular el error cuadrático
                error += (I_ratio - y_pred)**2
            return error

        # Optimizar factores de escala
        initial_guess = np.ones(n_fases)
        result = minimize(loss, initial_guess, method='L-BFGS-B')
        scaling_factors = result.x

        # Escalar y combinar patrones
        patterns = [np.array(phase[0]) * scaling_factors[i] for i, phase in enumerate(selected_fases)]
        mixPatt = np.sum(patterns, axis=0)
        max_val = mixPatt.max()
        norm_pattern = (100 * mixPatt / max_val) if max_val > 0 else np.zeros_like(mixPatt)

        # Etiquetas (proporciones lineales)
        mixQuants = np.sum([np.array(phase[1]) * prop for phase, prop in zip(selected_fases, props)], axis=0).tolist()

        miXRDs.append((norm_pattern.tolist(), mixQuants))

        #print(mixQuants)
        #plt.plot(norm_pattern)
        #plt.show()
        # Guardar si es necesario (código existente)
        if save:
            IDs = []
            for mineral in selected_fases:
                for i, j in enumerate(mineral[1]):
                    if j == 100:  # Encuentra el índice del mineral dominante
                        IDs.append(i)

            # Construir un nombre único para el archivo JSON
            file_name = "_".join(
                f"{IDToMin[IDs[i]]}_{proporciones[i]*100:.2f}" for i in range(n_fases)
            )
            file_name = f"{save_path}{file_name}.json"

            # Asegurar que no haya colisiones en los nombres de archivo
            suffix = 1
            while os.path.exists(file_name):
                file_name = f"{file_name.rstrip('.json')}_{suffix}.json"
                suffix += 1

            # Crear el contenido a guardar en el archivo JSON
            data_to_save = {
                "xrd": norm_pattern.tolist(),
                "q": mixQuants,
            }

            # Guardar el patrón mixto en un archivo JSON
            with open(file_name, 'w') as json_file:
                json.dump(data_to_save, json_file, indent=4)
            pass

    return miXRDs



Test=r"D:\Desktop\U\Tesis\Datos\Test"
testSave=r"D:\Desktop\U\Tesis\Datos\Test\Generated/"
QzCif=r"D:\Desktop\U\Tesis\Datos\cifs\Quartz"
PyCif=r"D:\Desktop\U\Tesis\Datos\cifs\Pyrite"
CalCif=r"D:\Desktop\U\Tesis\Datos\cifs\Calcite"
MgtCif=r"D:\Desktop\U\Tesis\Datos\cifs\Magnetite"
OCif=r"D:\Desktop\U\Tesis\Datos\cifs\Otros"
saves=r"D:\Desktop\U\Tesis\Datos\XRD\Synthetic/"
ev=r"D:\Desktop\U\Tesis\Datos\XRD\Synthetic\Evaluation/"
ky=r"D:\Desktop\U\Tesis\Datos\cifs"

#qz=savePatts(QzCif, patts_per_cif=4)
#py=savePatts(PyCif, patts_per_cif=22, save=True, save_path=testSave)
#cal=savePatts(CalCif, patts_per_cif=3)
#mgt=savePatts(MgtCif, patts_per_cif=2)
#other=savePatts(OCif, patts_per_cif=5)
ev=savePatts(ky,patts_per_cif=5, save=True, save_path=ev)

#xrd1=qz+py+cal+mgt
#xrd1=qz+py+cal+mgt+other

#mix1=mixPatts(xrd1, num_of_patts=1000, save=True, save_path=saves)+mixPatts(xrd1, n_fases=3, num_of_patts=1500, save=True, save_path=saves)+mixPatts(xrd1, n_fases=4, num_of_patts=2000, save=True, save_path=saves)+mixPatts(xrd1, n_fases=5, num_of_patts=2500, save=True, save_path=saves)
#mix2=mixPatts(xrd2, num_of_patts=800, save=True, save_path=saves)+mixPatts(xrd2, n_fases=3, num_of_patts=1000, save=True, save_path=saves)+mixPatts(xrd2, n_fases=4, num_of_patts=1200, save=True, save_path=saves)+mixPatts(xrd2, n_fases=5, num_of_patts=1400, save=True, save_path=saves)

#xrd1=savePatts(Test)
#xrd2=savePatts(Test)
#xrd=xrd1+xrd2

#mix=mixPatts(xrd, n_fases=3, num_of_patts=5, save=True, save_path=testSave)
#mix=mixPatts(xrd, n_fases=3, num_of_patts=2, save=True, save_path=testSave)
