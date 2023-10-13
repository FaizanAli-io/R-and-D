import cv2
import sympy
import logging
import numpy as np

from Ransac.coping_data import (
    CIR_copings,
    COR_copings,
)


def distance(pt1, pt2):
    pt1 = np.squeeze(np.asarray(pt1, dtype="float"))
    pt2 = np.squeeze(np.asarray(pt2, dtype="float"))
    return np.linalg.norm(pt1 - pt2)


def find_point_at_distance(points, current_index, target_distance):
    # Compute the Euclidean distance for points from current_index + 1 onwards
    distances = np.sqrt(np.sum((points[current_index] - points[current_index + 1:]) ** 2, axis=1))

    # Find the indices of the points that are close to the target_distance
    # We use np.isclose for floating point comparison with a tolerance

    # max atol : 10
    # min atol : 3 (not stable)
    indices = np.where(np.isclose(distances, target_distance, atol=10))

    # If indices are found, iterate over them to find the next closest point
    # where the index difference is less than half of the array size
    if indices[0].size:
        for index in indices[0]:
            closest_index = current_index + index + 1
            if abs(closest_index - current_index) <= len(points) / 2:
                return closest_index

    # If no suitable point is found, return None
    return None


def find_distance_at_point(points, current_index, target_index):
    return np.sqrt(np.sum((points[current_index] - points[target_index]) ** 2, axis=0))


def get_interp_type(i, ranges):
    for t, s, e in ranges:
        if s <= i <= e:
            return t


def get_last_linear(i, ranges):
    for _, s, e in ranges:
        if s <= i <= e:
            return e


def get_line_segment(a, ranges, points, scale):
    if get_interp_type(a, ranges) != "line":
        return None, None

    b = find_point_at_distance(points, a, 24 * scale)

    if not b:
        return None, None

    if get_interp_type(b, ranges) == "line":
        return b, "CS24-12"

    else:
        b = get_last_linear(a, ranges)
        dist = find_distance_at_point(points, a, b)

        if dist < 4 * scale:
            return b, "CS4-12"

        else:
            return b, "CS24-12"


def finding_outer_points(coping_type, ex1_x, ex1_y, mp_x, mp_y, c_x, c_y,
                         CAN_arr, PIXELPERINCH):

    if coping_type[:2] == "CS":
        logging.info("Coping Type: Straight")
        if ex1_x - mp_x == 0:
            gradient_f = float("inf")
        else:
            gradient_f = (ex1_y - mp_y) / (ex1_x - mp_x)

        if gradient_f == float("inf"):
            p_gradient_f = 0
        elif gradient_f == 0:
            p_gradient_f = float("inf")
        else:
            p_gradient_f = -1 / gradient_f

        if p_gradient_f == float("inf"):
            p_c1 = ex1_x
            p_c2 = mp_x
            x, y = sympy.symbols("x,y")
            eq1 = sympy.Eq(
                (x - ex1_x) ** 2 + (y - ex1_y) ** 2, (12 * PIXELPERINCH) ** 2
            )

            eq2 = sympy.Eq(x, p_c1)

            eq3 = sympy.Eq((x - mp_x) ** 2 + (y - mp_y) ** 2, (12 * PIXELPERINCH) ** 2)
            eq4 = sympy.Eq(x, p_c2)

            result_1 = sympy.solve((eq1, eq2), (x, y), dict=True)
            result_2 = sympy.solve((eq3, eq4), (x, y), dict=True)

            rs1_1, rs1_2 = (int(result_1[0][x]), int(result_1[0][y])), (
                int(result_1[1][x]),
                int(result_1[1][y]),
            )
            rs2_1, rs2_2 = (int(result_2[0][x]), int(result_2[0][y])), (
                int(result_2[1][x]),
                int(result_2[1][y]),
            )
        elif p_gradient_f == 0:
            p_c1 = ex1_y
            p_c2 = mp_y
            x, y = sympy.symbols("x,y")
            eq1 = sympy.Eq(
                (x - ex1_x) ** 2 + (y - ex1_y) ** 2, (12 * PIXELPERINCH) ** 2
            )
            eq2 = sympy.Eq(y, p_c1)

            eq3 = sympy.Eq((x - mp_x) ** 2 + (y - mp_y) ** 2, (12 * PIXELPERINCH) ** 2)
            eq4 = sympy.Eq(y, p_c2)
            result_1 = sympy.solve((eq1, eq2), (x, y), dict=True)
            result_2 = sympy.solve((eq3, eq4), (x, y), dict=True)

            logging.info("[Coping Fitter] RESULT_1")

            rs1_1, rs1_2 = (int(result_1[0][x]), int(result_1[0][y])), (
                int(result_1[1][x]),
                int(result_1[1][y]),
            )
            rs2_1, rs2_2 = (int(result_2[0][x]), int(result_2[0][y])), (
                int(result_2[1][x]),
                int(result_2[1][y]),
            )
        else:
            p_c1 = ex1_y - p_gradient_f * ex1_x
            p_c2 = mp_y - p_gradient_f * mp_x

            x, y = sympy.symbols("x,y")
            eq1 = sympy.Eq(
                (x - ex1_x) ** 2 + (y - ex1_y) ** 2,
                (12 * PIXELPERINCH) ** 2,
            )
            eq2 = sympy.Eq(y, p_gradient_f * x + p_c1)

            eq3 = sympy.Eq((x - mp_x) ** 2 + (y - mp_y) ** 2, (12 * PIXELPERINCH) ** 2)
            eq4 = sympy.Eq(y, p_gradient_f * x + p_c2)
            result_1 = sympy.solve((eq1, eq2), (x, y), dict=True)
            result_2 = sympy.solve((eq3, eq4), (x, y), dict=True)

            rs1_1, rs1_2 = (int(result_1[0][x]), int(result_1[0][y])), (
                int(result_1[1][x]),
                int(result_1[1][y]),
            )

            rs2_1, rs2_2 = (int(result_2[0][x]), int(result_2[0][y])), (
                int(result_2[1][x]),
                int(result_2[1][y]),
            )

        logging.info("[Coping Fitter] RESULT_1")
        logging.info(str(result_1))

        logging.info("[Coping Fitter] RESULT_2")
        logging.info(str(result_2))

        if cv2.pointPolygonTest(CAN_arr.astype(dtype=np.int32), rs1_1, True) < 0:
            ex1_outer_x, ex1_outer_y = rs1_1
        else:
            ex1_outer_x, ex1_outer_y = rs1_2

        if cv2.pointPolygonTest(CAN_arr.astype(dtype=np.int32), rs2_1, True) < 0:
            mp_outer_x, mp_outer_y = rs2_1
        else:
            mp_outer_x, mp_outer_y = rs2_2

    elif coping_type[:3] == "CIR":
        logging.info("Coping Type: CIR")
        dist = distance((ex1_x, ex1_y), (c_x, c_y))
        t = ((12 * PIXELPERINCH) + dist) / dist
        ex1_outer_x, ex1_outer_y = ((t * ex1_x) + ((1 - t) * c_x)), (
            (t * ex1_y) + ((1 - t) * c_y)
        )
        dist = distance((mp_x, mp_y), (c_x, c_y))
        t = ((12 * PIXELPERINCH) + dist) / dist
        mp_outer_x, mp_outer_y = ((t * mp_x) + ((1 - t) * c_x)), (t * mp_y) + (
            (1 - t) * c_y
        )

    elif coping_type[:3] == "COR":
        logging.info("Coping Type: COR")
        dist = distance((ex1_x, ex1_y), (c_x, c_y))
        t = (dist - (12 * PIXELPERINCH)) / dist
        ex1_outer_x, ex1_outer_y = ((t * ex1_x) + ((1 - t) * c_x)), (
            (t * ex1_y) + ((1 - t) * c_y)
        )
        dist = distance((mp_x, mp_y), (c_x, c_y))
        t = (dist - (12 * PIXELPERINCH)) / dist
        mp_outer_x, mp_outer_y = ((t * mp_x) + ((1 - t) * c_x)), (t * mp_y) + (
            (1 - t) * c_y
        )

    return ex1_outer_x, ex1_outer_y, mp_outer_x, mp_outer_y


def get_look_ahead_distance(scale):
    chordlengths = [row[2] for row in CIR_copings + COR_copings]
    return np.mean(chordlengths) * scale


def add_coping(dic, p1, p2, p3, p4, cp, r, ct):
    dic.append({
        "starting_point": [int(p1[0]), int(p1[1])],
        "ending_point": [int(p2[0]), int(p2[1])],
        "starting_outer_point": [int(p3[0]), int(p3[1])],
        "ending_outer_point": [int(p4[0]), int(p4[1])],
        "center": [None, None] if not cp else [int(cp[0]), int(cp[1])],
        "radius": None if not r else int(r),
        "coping": ct,
    })

    return dic


def coping_display(dic):
    for entry in dic:
        print(entry)
