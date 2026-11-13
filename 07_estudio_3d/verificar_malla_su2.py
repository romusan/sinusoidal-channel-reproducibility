"""Comprueba los elementos, el contorno y la coincidencia de una malla SU2."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def field(line, key):
    name, value = line.split("=", maxsplit=1)
    require(name.strip() == key, f"Se esperaba {key}, aparece {name.strip()}")
    return value.strip()


def rows(lines, count, length, dtype):
    data = np.array([next(lines).split() for _ in range(count)], dtype=dtype)
    require(data.shape == (count, length), "Longitud inesperada de fila")
    return data


def read_zone(lines, first):
    require(field(first, "NDIME") == "3", "Se requiere una malla 3D")
    cells = rows(lines, int(field(next(lines), "NELEM")), 10, np.int64)
    require((cells[:, 0] == 12).all(), "Hay elementos que no son hexaedros")
    require(np.array_equal(cells[:, -1], np.arange(len(cells))), "Índices de celda no consecutivos")
    cells = cells[:, 1:-1]
    points = rows(lines, int(field(next(lines), "NPOIN")), 3, float)
    require(np.isfinite(points).all(), "Se encontraron coordenadas no finitas")
    require(cells.min() >= 0 and cells.max() < len(points), "Celda con índices fuera de rango")
    markers = {}
    for _ in range(int(field(next(lines), "NMARK"))):
        marker = field(next(lines), "MARKER_TAG")
        require(marker not in markers, f"Marcador duplicado {marker}")
        faces = rows(lines, int(field(next(lines), "MARKER_ELEMS")), 5, np.int64)
        require((faces[:, 0] == 9).all(), f"Hay caras no cuadrangulares en {marker}")
        markers[marker] = faces[:, 1:]
    return points, cells, markers


def verify_zone(points, cells, markers):
    ownership = {}
    counts = Counter()
    for local in HEX_FACES:
        for index, face in enumerate(cells[:, local]):
            key = tuple(sorted(face))
            counts[key] += 1
            ownership[key] = index
    require(max(counts.values()) <= 2, "Una cara tiene más de dos celdas adyacentes")
    expected_boundary = {face for face, count in counts.items() if count == 1}
    actual_boundary = set()
    for name, faces in markers.items():
        for face in faces:
            key = tuple(sorted(face))
            require(key in expected_boundary, f"Cara inexistente o interior en {name}")
            require(key not in actual_boundary, f"Cara de frontera repetida en {name}")
            actual_boundary.add(key)
            p = points[face]
            outward = p.mean(axis=0) - points[cells[ownership[key]]].mean(axis=0)
            normal = np.cross(p[1] - p[0], p[3] - p[0])
            require(np.dot(normal, outward) > 0, f"Orientación incorrecta de cara en {name}")
    require(actual_boundary == expected_boundary, "Faltan caras externas en los marcadores")

    # Jacobiano en los ocho vértices del hexaedro VTK 12.
    signs = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    p = points[cells]
    determinants = []
    for i, corner in enumerate(signs):
        derivatives = []
        for axis in range(3):
            neighbor = corner.copy()
            neighbor[axis] = 1 - neighbor[axis]
            j = np.flatnonzero((signs == neighbor).all(axis=1))[0]
            derivatives.append((p[:, j] - p[:, i]) * (1 - 2 * corner[axis]))
        determinants.append(np.einsum("ij,ij->i", np.cross(*derivatives[:2]), derivatives[2]))
    min_jacobian = float(np.min(determinants))
    require(min_jacobian > 0, "La malla contiene hexaedros invertidos o degenerados")
    return {"celdas": len(cells), "nodos": len(points),
            "caras_contorno": len(expected_boundary), "jacobiano_minimo_m3": min_jacobian}


def verify(path):
    zones = []
    with path.open(encoding="ascii") as stream:
        lines = iter(line.strip() for line in stream if line.strip() and not line.startswith("%"))
        first = next(lines)
        if first.startswith("NZONE="):
            require(field(first, "NZONE") == "2", "Se requieren dos zonas")
            for index in (1, 2):
                require(field(next(lines), "IZONE") == str(index), "Zona con índice inesperado")
                zones.append(read_zone(lines, next(lines)))
        else:
            zones.append(read_zone(lines, first))
        require(next(lines, None) is None, "Hay datos inesperados al final de la malla")
    summary = [verify_zone(*zone) for zone in zones]
    if len(zones) == 2:
        air, _, air_markers = zones[0]
        solid, _, solid_markers = zones[1]
        air_faces = air[air_markers["interface_air"]]
        solid_faces = solid[solid_markers["interface_solid"]]
        require(np.array_equal(air_faces, solid_faces[:, [0, 3, 2, 1]]),
                "Interfaz de CHT no coincidente con normales opuestas")
        require(all(len(faces) > 0 for faces in solid_markers.values()), "Marcador sólido vacío")

    sidecar = path.with_suffix(".npz")
    if sidecar.is_file():
        with np.load(sidecar) as structured:
            for index, (points, _, _) in enumerate(zones):
                name = ("air_xyz", "solid_xyz")[index]
                require(np.allclose(structured[name].reshape(-1, 3), points, rtol=0, atol=1e-15),
                        f"El orden nodal de {name} difiere del SU2")
    return {"archivo": path.name, "zonas": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mallas", nargs="+", type=Path)
    for mesh in parser.parse_args().mallas:
        print(json.dumps(verify(mesh), ensure_ascii=False))
