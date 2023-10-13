import math

PI = 3.14159265359

data = {
    "CIR_input": (
        ("CIR1", 12, 8.49, 8.49, 16+11/16),
        ("CIR2", 24, 10.99, 4+13/16, 18+7/8),
        ("CIR3", 36, 11.52, 3.36, 20+3/16),
        ("CIR4", 48, 11.71, 2.62, 20+15/16),
        ("CIR5", 60, 11.81, 2.14, 21+7/16),
        ("CIR6", 72, 11.86, 1.82, 21+13/16),
        ("CIR7", 84, 11.9, 1.57, 22),
        ("CIR8", 96, 11.92, 1.39, 22+3/16),
        ("CIR9", 108, 11.94, 1.24, 22+3/8),
        ("CIR10", 120, 11.95, 1+1/8, 22+1/2),
        ("CIR11", 132, 11.96, 1.03, 22+9/16),
        ("CIR12", 144, 11.96, 0.95, 22+11/16),
        ("CIR15", 180, 11.98, 0.76, 22+7/8),
        ("CIR17", 204+1/8, 11.98, 0.67, 22+15/16),
        ("CIR18", 216, 11.98, 0.64, 23),
        ("CIR20", 240, 11.986, 0.577, 23+1/16),
        ("CIR25", 300, 11.990, 1/2, 25),
        ("CIR35", 420, 11.995, 0.333, 23+5/16),
        ("CIR50", 600, 11.9977, 0.2338, 23+3/8),
    ),

    "CIR_old": (
        ('CIR1',12, 16.6875, 1.5377092353913122, 18.452510824695747),
        ('CIR2',24, 18.875, 0.8082821316614327, 19.398771159874386),
        ('CIR3',36, 20.1875, 0.5683839822039942, 20.461823359343793),
        ('CIR4',48, 20.9375, 0.4397322202410318, 21.107146571569526),
        ('CIR5',60, 21.4375, 0.35921994420884973, 21.553196652530985),
        ('CIR6',72, 21.8125, 0.30412204776629864, 21.8967874391735),
        ('CIR7',84, 22, 0.26265914475061924, 22.06336815905202),
        ('CIR8',96, 22.1875, 0.23163730672755659, 22.237181445845433),
        ('CIR9',108, 22.375, 0.20754824225753885, 22.415210163814194),
        ('CIR10',120, 22.5, 0.18777575021503295, 22.533090025803954),
        ('CIR11',132, 22.5625, 0.17113679639263762, 22.590057123828167),
        ('CIR12',144, 22.6875, 0.1577154926424745, 22.711030940516327),
        ('CIR15',180, 22.875, 0.12716900649624716, 22.890421169324487),
        ('CIR17',204.125, 22.9375, 0.11242907608574147, 22.949585156001977),
        ('CIR18',216, 23, 0.10653185074068368, 23.010879759987674),
        ('CIR20',240, 23.0625, 0.09613076056594638, 23.071382535827134),
        ('CIR25',300, 25, 0.08335746484515573, 25.00723945354672),
        ('CIR35',420, 23.3125, 0.05551308022197957, 23.31549369323142),
        ('CIR50',600, 23.375, 0.038960797465685426, 23.376478479411254)
    ),

    "COR_input": (
        ("COR1", 12, 8.49, 8.49, 16.97),
        ("COR2", 24, 8.49, 8.49, 33+11/16),
        ("COR3", 36, 10.99, 4+13/16, 28+7/16),
        ("COR4", 48, 11.5, 3+7/16, 27+1/16),
        ("COR5", 60, 11.7, 2+11/16, 26+5/16),
        ("COR6", 72, 11.8, 2+3/16, 25+13/16),
        ("COR7", 84, 11.85, 1+7/8, 25+1/2),
        ("COR8", 96, 11.89, 1+5/8, 25+1/4),
        ("COR9", 108, 11.91, 1+7/16, 25+1/16),
        ("COR10", 120, 11.93, 1+1/4, 24+7/8),
        ("COR11", 132, 11.95, 1+1/8, 24+3/4),
        ("COR12", 144, 11.95, 1+1/16, 24+11/16),
        ("COR15", 180, 11.96, 1, 24+7/16),
        ("COR17", 204, 11.98, 3/4, 24+3/8),
        ("COR18", 216, 11.98, 11/16, 24+5/16),
        ("COR20", 240, 11.98, 5/8, 24+1/4),
        ("COR25", 300, 11.990, 1/2, 24+1/8),
        ("COR35", 420, 11.990, 3/8, 24),
        ("COR50", 600, 11.997, 1/4, 23+7/8),
    ),

    "COR_old": (
        ('COR1',12, 16.79, 1.5496284265670792, 18.595541118804952),
        ('COR2',24, 33.6875, 1.5559065893315844, 37.341758143958025),
        ('COR3',36, 28.4375, 0.8120601112719208, 29.234164005789147),
        ('COR4',48, 27.0625, 0.5715498688825466, 27.434393706362236),
        ('COR5',60, 26.3125, 0.44213410679415993, 26.528046407649597),
        ('COR6',72, 25.8125, 0.36045516058331734, 25.952771561998848),
        ('COR7',84, 25.5, 0.3047493407044602, 25.598944619174656),
        ('COR8',96, 25.25, 0.2637849524453459, 25.323355434753203),
        ('COR9',108, 25.0625, 0.23258406899785908, 25.11907945176878),
        ('COR10',120, 24.875, 0.2076646093696004, 24.91975312435205),
        ('COR11',132, 24.75, 0.18777575021503295, 24.78639902838435),
        ('COR12',144, 24.6875, 0.17165162761526828, 24.71783437659863),
        ('COR15',180, 24.4375, 0.13586837141217234, 24.45630685419102),
        ('COR17',204, 24.375, 0.11955648601823073, 24.389523147719068),
        ('COR18',216, 24.3125, 0.11261737298984499, 24.32535256580652),
        ('COR20',240, 24.25, 0.10108469847006421, 24.26032763281541),
        ('COR25',300, 24.125, 0.08043835085183397, 24.13150525555019),
        ('COR35',420, 24, 0.05715063453858531, 24.00326650620583),
        ('COR50',600, 23.875, 0.03979429235104354, 23.876575410626128)
    ),
}

output = {
    "CIR_new": [],
    "CIR_diff_abs": [],
    "CIR_diff_per": [],
    "COR_new": [],
    "COR_diff_abs": [],
    "COR_diff_per": [],
}

for tiletype in ["CIR", "COR"]:

    for i, (n, r, a, b, l) in enumerate(data[f"{tiletype}_input"]):
        t = PI - 2 * math.atan(a / b)
        a = r * t
        output[f"{tiletype}_new"].append((n, r, l, t, a))

        _, pr, pl, pt, pa = data[f"{tiletype}_old"][i]
        dr, dl, dt, da = r-pr, l-pl, t-pt, a-pa
        output[f"{tiletype}_diff_abs"].append((n, round(dr, 2), round(dl, 2), round(dt, 2), round(da, 2)))
        output[f"{tiletype}_diff_per"].append((n, round(100*dr/r, 2), round(100*dl/l, 2), round(100*dt/t, 2), round(100*da/a, 2)))

    print(f"NEW {tiletype} DATASET")
    for row in output[f"{tiletype}_new"]:
        print(row)
    print()

    print("ABSOLUTE DIFF WITH OLD DATASET")
    for row in output[f"{tiletype}_diff_abs"]:
        print(row)
    print()

    print("PERCENTAGE DIFF WITH OLD DATASET")
    for row in output[f"{tiletype}_diff_per"]:
        print(row)
    print()