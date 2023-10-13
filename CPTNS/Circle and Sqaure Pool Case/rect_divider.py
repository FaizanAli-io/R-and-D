def rectangle_coping_fit(
        final_contours, ranges, temp, PIXELPERINCH, url, model):

    logging.info("[rectangle_coping_fit] started")
    final_coping_fit = list()

    ordered_points, borders = model.compute_square()
    ordered_points = np.array(ordered_points)

    cur = 0
    for i in range(len(borders)):
        tiles = 0
        sim_cur = cur
        while sim_cur < borders[i]:
            end = util.find_point_at_distance(
                ordered_points,
                sim_cur,
                24 * PIXELPERINCH,
            )

            tiles += 1

            if not end:
                break
            else:
                sim_cur = end + 1

        extra_space = 24 * PIXELPERINCH - math.dist(
            ordered_points[sim_cur], ordered_points[borders[i] - 1])
        diff_per_tile = (extra_space * 1.05) / tiles
        new_length = 24 * PIXELPERINCH - diff_per_tile

        logging.info(f"TILES PLACED: {tiles}")
        logging.info(f"EXTRA SPACE: {extra_space}")
        logging.info(f"DIFF / TILE: {diff_per_tile}")
        logging.info(f"NEW LENGTH: {new_length}")

        for j in range(tiles):
            end = util.find_point_at_distance(
                ordered_points,
                cur,
                new_length,
            )

            if not end:
                end = borders[i]

            p1x, p1y = ordered_points[cur]
            p2x, p2y = ordered_points[end]
            p3x, p3y, p4x, p4y = util.finding_outer_points(
                "CS24-12", p1x, p1y, p2x, p2y, None, None,
                ordered_points, PIXELPERINCH)

            final_coping_fit.append(
                {
                    "coping": "CS24-12",
                    "starting_point": [int(p1x), int(p1y)],
                    "ending_point": [int(p2x), int(p2y)],
                    "center": [None, None],
                    "radius": None,
                    "starting_outer_point": [int(p3x), int(p3y)],
                    "ending_outer_point": [int(p4x), int(p4y)],
                }
            )

            cur = end + 1

        cur = borders[i] + 1

    if DEBUG:
        plot_final_copings(temp, ordered_points, final_coping_fit)

    return json.dumps(
        {
            "copings": final_coping_fit,
            "scale": PIXELPERINCH,
            "height": temp.shape[0],
            "width": temp.shape[1],
            "url": url,
        }
    )