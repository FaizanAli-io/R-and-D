import cv2
import json
import logging

import numpy as np

from Ransac import ransac_util2 as util
from Ransac import ransac_debug as debug

from Ransac.LSC import LSC

from Ransac.coping_data import (
    CIR_copings,
    COR_copings,
)

DEBUG = False


def tile_placer(contours, ranges, scale, img, url):
    final_coping = []
    current_index = 0
    total = len(contours)
    curved_range = util.get_look_ahead_distance(scale)

    logging.info(f"Number of Points: {total}")
    logging.info(f"Lookahead range: {curved_range}")
    logging.info(f"shape: {contours.shape}")

    if DEBUG:
        debug.plot_interpolated(img, contours)

    while True:
        logging.info(f"Current Index: {current_index}")

        # Straight Line Logic

        ending_point_index, coping_type = util.get_line_segment(
            current_index, ranges, contours, scale)

        if ending_point_index:
            p1x, p1y = contours[current_index]
            p2x, p2y = contours[ending_point_index]

            p3x, p3y, p4x, p4y = util.finding_outer_points(
                coping_type, p1x, p1y, p2x, p2y, None, None, contours, scale)

            final_coping = util.add_coping(
                final_coping,
                ct=coping_type,
                p1=(p1x, p1y),
                p2=(p2x, p2y),
                p3=(p3x, p3y),
                p4=(p4x, p4y),
                cp=None,
                r=None,
            )

        # Curved Tile Logic

        else:
            range_end_index = util.find_point_at_distance(
                contours, current_index, curved_range)

            curved_section = contours[current_index:range_end_index]

            logging.info(f"RANGE: {current_index}:{range_end_index}")

            algo = LSC(curved_section, scale)
            cx, cy, cr = algo.compute()

            midpoint = (
                int((contours[current_index][0] +
                    contours[current_index][1]) / 2),
                int((contours[range_end_index][0] +
                    contours[range_end_index][1]) / 2),
            )

            cval = cv2.pointPolygonTest(
                contours.astype(dtype=np.int32), midpoint, True)

            logging.info(f"CR: {cr}")
            logging.info(f"CVAL: {cval}")
            logging.info(f"CENTER: ({cx}, {cy})")

            coping = algo.choose(cr, CIR_copings if cval <= 0 else COR_copings)
            coping_type = coping[0]
            coping_length = coping[2]

            ending_point_index = util.find_point_at_distance(
                contours, current_index, coping_length * scale)

            p1x, p1y = contours[current_index]
            p2x, p2y = contours[ending_point_index]

            p3x, p3y, p4x, p4y = util.finding_outer_points(
                coping_type, p1x, p1y, p2x, p2y, cx, cy, contours, scale)

            final_coping = util.add_coping(
                final_coping,
                ct=coping_type,
                p1=(p1x, p1y),
                p2=(p2x, p2y),
                p3=(p3x, p3y),
                p4=(p4x, p4y),
                cp=(cx, cy),
                r=cr,
            )

        current_index = ending_point_index + 1
        logging.info(f"Coping Type: {coping_type}")

        if DEBUG:
            debug.plot_inner_and_outer_points(
                img,
                p1x,
                p1y,
                p2x,
                p2y,
                p3x,
                p3y,
                p4x,
                p4y,
            )

        if current_index >= total * 0.9:
            break

    dic = {
        "copings": final_coping,
        "scale": scale,
        "height": img.shape[0],
        "width": img.shape[1],
        "url": url,
    }

    return json.dumps(dic)
