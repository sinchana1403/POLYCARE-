import numpy as np

# FEATURES 
FEATURE_NAMES = [
    "BMI",                    # 0
    "LH(mIU/mL)",             # 1
    "FSH(mIU/mL)",            # 2
    "FSH/LH",                 # 3 
    "AMH(ng/mL)",             # 4  
    "Waist:Hip Ratio",        # 5  
    "Follicle No. (L)",       # 6
    "Follicle No. (R)",       # 7
    "Avg. F size (L) (mm)",   # 8
    "Avg. F size (R) (mm)",   # 9
    "Weight gain(Y/N)",       # 10
    "hair growth(Y/N)",       # 11
    "Skin darkening (Y/N)",   # 12
    "Pimples(Y/N)",           # 13
    "Fast food (Y/N)",        # 14
    "Reg.Exercise(Y/N)",      # 15
    "Age (yrs)",              # 16
    "Weight (Kg)",            # 17
    "Cycle(R/I)",             # 18  
    "Cycle length(days)",     # 19  
    "TSH (mIU/L)",            # 20
    "PRL(ng/mL)",             # 21
    "Vit D3 (ng/mL)",         # 22
    "PRG(ng/mL)",             # 23
    "RBS(mg/dl)",             # 24
    "BP _Systolic (mmHg)",    # 25
    "BP _Diastolic (mmHg)",   # 26
    "Endometrium (mm)",       # 27
    "Hb(g/dl)",               # 28
    "Pulse rate(bpm)",        # 29
]

# normal ranges that is displayed on our form 
NORMAL_RANGES = {
    "BMI":                   (18.5, 24.9,  "kg/m²"),
    "LH(mIU/mL)":            (2.0,  15.0,  "mIU/mL"),
    "FSH(mIU/mL)":           (3.0,  10.0,  "mIU/mL"),
    "AMH(ng/mL)":            (1.0,  3.5,   "ng/mL"),
    "Waist:Hip Ratio":       (0.0,  0.85,  "ratio"),
    "Follicle No. (L)":      (0.0,  12.0,  "count"),
    "Follicle No. (R)":      (0.0,  12.0,  "count"),
    "Avg. F size (L) (mm)":  (0.0,  10.0,  "mm"),
    "Avg. F size (R) (mm)":  (0.0,  10.0,  "mm"),
    "TSH (mIU/L)":           (0.4,  4.0,   "mIU/L"),
    "PRL(ng/mL)":            (2.0,  29.0,  "ng/mL"),
    "RBS(mg/dl)":            (70.0, 140.0, "mg/dL"),
    "BP _Systolic (mmHg)":   (90.0, 120.0, "mmHg"),
    "BP _Diastolic (mmHg)":  (60.0, 80.0,  "mmHg"),
    "Hb(g/dl)":              (12.0, 16.0,  "g/dL"),
}

DISPLAY_NAMES = {
    "BMI":                   "BMI",
    "LH(mIU/mL)":           "LH(mIU/mL)",
    "FSH(mIU/mL)":          "FSH(mIU/mL)",
    "AMH(ng/mL)":           "AMH(ng/mL)",
    "Waist:Hip Ratio":      "Waist:Hip Ratio",
    "Follicle No. (L)":     "Follicle No. (L)",
    "Follicle No. (R)":     "Follicle No. (R)",
    "Avg. F size (L) (mm)": "Avg. F size L (mm)",
    "Avg. F size (R) (mm)": "Avg. F size R (mm)",
    "TSH (mIU/L)":          "TSH (mIU/L)",
    "PRL(ng/mL)":           "PRL(ng/mL)",
    "RBS(mg/dl)":           "RBS(mg/dl)",
    "BP _Systolic (mmHg)":  "BP Systolic (mmHg)",
    "BP _Diastolic (mmHg)": "BP Diastolic (mmHg)",
    "Hb(g/dl)":             "Hb(g/dl)",
}
MEDICAL_INPUT_BOUNDS = {
    # field_key          (min,   max,   human_label,               unit)
    'age':           (10,    70,    'Age',                     'years'),
    'weight':        (20,    200,   'Weight',                  'kg'),
    'height':        (100,   220,   'Height',                  'cm'),
    'waist':         (40,    200,   'Waist circumference',     'cm'),
    'hip':           (50,    200,   'Hip circumference',       'cm'),
    'LH':            (0.1,   150,   'LH',                     'mIU/mL'),
    'FSH':           (0.1,   150,   'FSH',                    'mIU/mL'),
    'AMH':           (0.01,  50,    'AMH',                    'ng/mL'),
    'TSH':           (0.01,  100,   'TSH',                    'mIU/L'),
    'PRL':           (0.1,   500,   'Prolactin',               'ng/mL'),
    'PRG':           (0.01,  100,   'Progesterone',            'ng/mL'),
    'RBS':           (50,    600,   'Blood glucose (RBS)',     'mg/dL'),
    'vitD3':         (1,     200,   'Vitamin D3',              'ng/mL'),
    'bp_systolic':   (60,    220,   'Systolic BP',             'mmHg'),
    'bp_diastolic':  (40,    150,   'Diastolic BP',            'mmHg'),
    'pulse_rate':    (30,    200,   'Pulse rate',              'bpm'),
    'hemoglobin':    (3,     25,    'Haemoglobin',             'g/dL'),
    'follicle_L':    (0,     50,    'Follicle count (Left)',   'count'),
    'follicle_R':    (0,     50,    'Follicle count (Right)',  'count'),
    'fsize_L':       (0,     40,    'Follicle size (Left)',    'mm'),
    'fsize_R':       (0,     40,    'Follicle size (Right)',   'mm'),
    'endometrium':   (1,     30,    'Endometrium thickness',   'mm'),
    'cycle_length':  (1,     90,    'Cycle length',            'days'),
}
_FEATURE_CLAMPS = {
    "BMI":                   (10.0, 70.0),
    "LH(mIU/mL)":            (0.1,  150.0),
    "FSH(mIU/mL)":           (0.1,  150.0),
    "FSH/LH":                (0.0,  50.0),
    "AMH(ng/mL)":            (0.07, 357.0),   
    "Waist:Hip Ratio":       (0.3,  2.0),
    "Follicle No. (L)":      (0,    50),
    "Follicle No. (R)":      (0,    50),
    "Avg. F size (L) (mm)":  (0.0,  40.0),
    "Avg. F size (R) (mm)":  (0.0,  40.0),
    "TSH (mIU/L)":           (0.01, 100.0),
    "PRL(ng/mL)":            (0.1,  500.0),
    "Vit D3 (ng/mL)":        (1.0,  200.0),
    "PRG(ng/mL)":            (0.01, 100.0),
    "RBS(mg/dl)":            (50.0, 600.0),
    "BP _Systolic (mmHg)":   (60.0, 220.0),
    "BP _Diastolic (mmHg)":  (40.0, 150.0),
    "Endometrium (mm)":      (1.0,  30.0),
    "Hb(g/dl)":              (3.0,  25.0),
    "Pulse rate(bpm)":       (30.0, 200.0),
    "Age (yrs)":             (10.0, 70.0),
    "Weight (Kg)":           (20.0, 200.0),
}
def _f(val, default=0.0):
    try:    return float(val)
    except: return default

def _i(val, default=0):
    try:    return int(float(val))
    except: return default

def _cycle_regularity_code(raw_value) -> int:
    return 2 if str(raw_value).strip().upper() == "I" else 4

def _cycle_length_code(days) -> int:
    d = _f(days, 28.0)
    if d <= 21: return 3
    if d <= 35: return 4
    return 5

def _amh_to_pmol(amh_ngml) -> float:
    return _f(amh_ngml, 0.0) * 7.14


# ─── Input Validation ────────────────────────────────────────────────────────
def validate_medical_input_ranges(data: dict) -> dict:
    errors = []

    for field, (lo, hi, label, unit) in MEDICAL_INPUT_BOUNDS.items():
        raw = data.get(field)
        if raw in (None, '', 'null'):
            continue  
        try:
            v = float(raw)
        except (ValueError, TypeError):
            errors.append(f"{label}: must be a numeric value (got '{raw}')")
            continue

        if v < lo or v > hi:
            errors.append(
                f"{label}: {v} {unit} is outside the valid range "
                f"({lo}–{hi} {unit})"
            )


import numpy as np

FEATURE_NAMES = [
    "BMI",                    # 0
    "LH(mIU/mL)",             # 1
    "FSH(mIU/mL)",            # 2
    "FSH/LH",                 # 3  
    "AMH(ng/mL)",             # 4  
    "Waist:Hip Ratio",        # 5  
    "Follicle No. (L)",       # 6
    "Follicle No. (R)",       # 7
    "Avg. F size (L) (mm)",   # 8
    "Avg. F size (R) (mm)",   # 9
    "Weight gain(Y/N)",       # 10
    "hair growth(Y/N)",       # 11
    "Skin darkening (Y/N)",   # 12
    "Pimples(Y/N)",           # 13
    "Fast food (Y/N)",        # 14
    "Reg.Exercise(Y/N)",      # 15
    "Age (yrs)",              # 16
    "Weight (Kg)",            # 17
    "Cycle(R/I)",             # 18  
    "Cycle length(days)",     # 19  
    "TSH (mIU/L)",            # 20
    "PRL(ng/mL)",             # 21
    "Vit D3 (ng/mL)",         # 22
    "PRG(ng/mL)",             # 23
    "RBS(mg/dl)",             # 24
    "BP _Systolic (mmHg)",    # 25
    "BP _Diastolic (mmHg)",   # 26
    "Endometrium (mm)",       # 27
    "Hb(g/dl)",               # 28
    "Pulse rate(bpm)",        # 29
]

NORMAL_RANGES = {
    "BMI":                   (18.5, 24.9,  "kg/m²"),
    "LH(mIU/mL)":            (2.0,  15.0,  "mIU/mL"),
    "FSH(mIU/mL)":           (3.0,  10.0,  "mIU/mL"),
    "AMH(ng/mL)":            (1.0,  3.5,   "ng/mL"),
    "Waist:Hip Ratio":       (0.0,  0.85,  "ratio"),
    "Follicle No. (L)":      (0.0,  12.0,  "count"),
    "Follicle No. (R)":      (0.0,  12.0,  "count"),
    "Avg. F size (L) (mm)":  (0.0,  10.0,  "mm"),
    "Avg. F size (R) (mm)":  (0.0,  10.0,  "mm"),
    "TSH (mIU/L)":           (0.4,  4.0,   "mIU/L"),
    "PRL(ng/mL)":            (2.0,  29.0,  "ng/mL"),
    "RBS(mg/dl)":            (70.0, 140.0, "mg/dL"),
    "BP _Systolic (mmHg)":   (90.0, 120.0, "mmHg"),
    "BP _Diastolic (mmHg)":  (60.0, 80.0,  "mmHg"),
    "Hb(g/dl)":              (12.0, 16.0,  "g/dL"),
}

DISPLAY_NAMES = {
    "BMI":                   "BMI",
    "LH(mIU/mL)":           "LH(mIU/mL)",
    "FSH(mIU/mL)":          "FSH(mIU/mL)",
    "AMH(ng/mL)":           "AMH(ng/mL)",
    "Waist:Hip Ratio":      "Waist:Hip Ratio",
    "Follicle No. (L)":     "Follicle No. (L)",
    "Follicle No. (R)":     "Follicle No. (R)",
    "Avg. F size (L) (mm)": "Avg. F size L (mm)",
    "Avg. F size (R) (mm)": "Avg. F size R (mm)",
    "TSH (mIU/L)":          "TSH (mIU/L)",
    "PRL(ng/mL)":           "PRL(ng/mL)",
    "RBS(mg/dl)":           "RBS(mg/dl)",
    "BP _Systolic (mmHg)":  "BP Systolic (mmHg)",
    "BP _Diastolic (mmHg)": "BP Diastolic (mmHg)",
    "Hb(g/dl)":             "Hb(g/dl)",
}

MEDICAL_INPUT_BOUNDS = {
    # field_key          (min,   max,   human_label,               unit)
    'age':           (10,    70,    'Age',                     'years'),
    'weight':        (20,    200,   'Weight',                  'kg'),
    'height':        (100,   220,   'Height',                  'cm'),
    'waist':         (40,    200,   'Waist circumference',     'cm'),
    'hip':           (50,    200,   'Hip circumference',       'cm'),
    'LH':            (0.1,   150,   'LH',                     'mIU/mL'),
    'FSH':           (0.1,   150,   'FSH',                    'mIU/mL'),
    'AMH':           (0.01,  50,    'AMH',                    'ng/mL'),
    'TSH':           (0.01,  100,   'TSH',                    'mIU/L'),
    'PRL':           (0.1,   500,   'Prolactin',               'ng/mL'),
    'PRG':           (0.01,  100,   'Progesterone',            'ng/mL'),
    'RBS':           (50,    600,   'Blood glucose (RBS)',     'mg/dL'),
    'vitD3':         (1,     200,   'Vitamin D3',              'ng/mL'),
    'bp_systolic':   (60,    220,   'Systolic BP',             'mmHg'),
    'bp_diastolic':  (40,    150,   'Diastolic BP',            'mmHg'),
    'pulse_rate':    (30,    200,   'Pulse rate',              'bpm'),
    'hemoglobin':    (3,     25,    'Haemoglobin',             'g/dL'),
    'follicle_L':    (0,     50,    'Follicle count (Left)',   'count'),
    'follicle_R':    (0,     50,    'Follicle count (Right)',  'count'),
    'fsize_L':       (0,     40,    'Follicle size (Left)',    'mm'),
    'fsize_R':       (0,     40,    'Follicle size (Right)',   'mm'),
    'endometrium':   (1,     30,    'Endometrium thickness',   'mm'),
    'cycle_length':  (1,     90,    'Cycle length',            'days'),
}


_FEATURE_CLAMPS = {
    "BMI":                   (10.0, 70.0),
    "LH(mIU/mL)":            (0.1,  150.0),
    "FSH(mIU/mL)":           (0.1,  150.0),
    "FSH/LH":                (0.0,  50.0),
    "AMH(ng/mL)":            (0.07, 357.0),   # pmol/L: (0.01 × 7.14, 50 × 7.14)
    "Waist:Hip Ratio":       (0.3,  2.0),
    "Follicle No. (L)":      (0,    50),
    "Follicle No. (R)":      (0,    50),
    "Avg. F size (L) (mm)":  (0.0,  40.0),
    "Avg. F size (R) (mm)":  (0.0,  40.0),
    "TSH (mIU/L)":           (0.01, 100.0),
    "PRL(ng/mL)":            (0.1,  500.0),
    "Vit D3 (ng/mL)":        (1.0,  200.0),
    "PRG(ng/mL)":            (0.01, 100.0),
    "RBS(mg/dl)":            (50.0, 600.0),
    "BP _Systolic (mmHg)":   (60.0, 220.0),
    "BP _Diastolic (mmHg)":  (40.0, 150.0),
    "Endometrium (mm)":      (1.0,  30.0),
    "Hb(g/dl)":              (3.0,  25.0),
    "Pulse rate(bpm)":       (30.0, 200.0),
    "Age (yrs)":             (10.0, 70.0),
    "Weight (Kg)":           (20.0, 200.0),
}


def _f(val, default=0.0):
    try:    return float(val)
    except: return default

def _i(val, default=0):
    try:    return int(float(val))
    except: return default

def _cycle_regularity_code(raw_value) -> int:
    """Frontend sends R/I string. Model expects 4=Regular, 2=Irregular."""
    return 2 if str(raw_value).strip().upper() == "I" else 4

def _cycle_length_code(days) -> int:
    """Convert actual days to Kaggle PCOS dataset category code.
    Kaggle encoding: 3=short (≤21d), 4=normal (22–35d), 5=long (>35d).
    Note: 35 days is used instead of 28 because WHO defines oligomenorrhea
    as cycles >35 days, which is itself a Rotterdam criterion for PCOS.
    """
    d = _f(days, 28.0)
    if d <= 21: return 3
    if d <= 35: return 4
    return 5

def _amh_to_pmol(amh_ngml) -> float:
    """Convert ng/mL to pmol/L (training data unit). 1 ng/mL = 7.14 pmol/L."""
    return _f(amh_ngml, 0.0) * 7.14


def validate_medical_input_ranges(data: dict) -> dict:
    errors = []

    for field, (lo, hi, label, unit) in MEDICAL_INPUT_BOUNDS.items():
        raw = data.get(field)
        if raw in (None, '', 'null'):
            continue  # optional field — skip, defaults applied later
        try:
            v = float(raw)
        except (ValueError, TypeError):
            errors.append(f"{label}: must be a numeric value (got '{raw}')")
            continue

        if v < lo or v > hi:
            errors.append(
                f"{label}: {v} {unit} is outside the valid range "
                f"({lo}–{hi} {unit})"
            )

    # checking feils like bp
    bp_sys = _f(data.get('bp_systolic'))
    bp_dia = _f(data.get('bp_diastolic'))
    if bp_sys and bp_dia and bp_sys <= bp_dia:
        errors.append(
            f"Systolic BP ({bp_sys} mmHg) must be greater than "
            f"Diastolic BP ({bp_dia} mmHg)"
        )

    waist = _f(data.get('waist'))
    hip   = _f(data.get('hip'))
    if waist and hip and (waist / hip) > 1.5:
        errors.append(
            f"Waist-to-Hip ratio ({waist / hip:.2f}) is physiologically "
            f"implausible — check waist ({waist} cm) and hip ({hip} cm)"
        )

    lh  = _f(data.get('LH'))
    fsh = _f(data.get('FSH'))
    if lh and fsh and (lh / fsh) > 20:
        errors.append(
            f"LH/FSH ratio ({lh/fsh:.1f}) appears implausible — "
            f"check LH ({lh}) and FSH ({fsh}) values"
        )

    return {'valid': len(errors) == 0, 'errors': errors}


# feature engineering 
def build_feature_vector(data: dict) -> np.ndarray:
    age    = _f(data.get("age"),    25.0)
    weight = _f(data.get("weight"), 60.0)
    height = _f(data.get("height"), 160.0)
    waist  = _f(data.get("waist"),  32.0)
    hip    = _f(data.get("hip"),    38.0)

    bmi = _f(data.get("BMI"), 0.0)
    if bmi <= 0 and height > 0:
        bmi = weight / ((height / 100.0) ** 2)
    bmi = round(bmi, 2)

    lh  = _f(data.get("LH"),  8.0)
    fsh = _f(data.get("FSH"), 6.0)
    tsh = _f(data.get("TSH"), 2.0)
    prl = _f(data.get("PRL"), 15.0)
    prg = _f(data.get("PRG"), 1.0)

    amh_pmol        = _amh_to_pmol(data.get("AMH", 0.0))
    fsh_lh_ratio    = round(fsh / lh,    4) if lh > 0.1 else 0.0
    waist_hip_ratio = round(waist / hip, 4) if hip > 0   else 0.0

    cycle_code     = _cycle_regularity_code(data.get("cycle", "R"))
    cycle_len_code = _cycle_length_code(data.get("cycle_length", 28))

    follicle_l  = _f(data.get("follicle_L"),  6.0)
    follicle_r  = _f(data.get("follicle_R"),  6.0)
    fsize_l     = _f(data.get("fsize_L"),    10.0)
    fsize_r     = _f(data.get("fsize_R"),    10.0)
    endometrium = _f(data.get("endometrium"),  7.0)
    bp_sys      = _f(data.get("bp_systolic"),  110.0)
    bp_dia      = _f(data.get("bp_diastolic"),  70.0)
    hemoglobin  = _f(data.get("hemoglobin"),   12.0)
    pulse_rate  = _f(data.get("pulse_rate"),   72.0)
    rbs         = _f(data.get("RBS"),          95.0)
    vit_d3      = _f(data.get("vitD3"),        25.0)

    weight_gain    = _i(data.get("weight_gain",    0))
    hair_growth    = _i(data.get("hair_growth",    0))
    skin_darkening = _i(data.get("skin_darkening", 0))
    pimples        = _i(data.get("pimples",        0))
    fast_food      = _i(data.get("fast_food",      0))
    exercise       = _i(data.get("exercise",       0))

    feature_map = {
        "BMI":                   bmi,
        "LH(mIU/mL)":           lh,
        "FSH(mIU/mL)":          fsh,
        "FSH/LH":               fsh_lh_ratio,
        "AMH(ng/mL)":           amh_pmol,
        "Waist:Hip Ratio":      waist_hip_ratio,
        "Follicle No. (L)":     follicle_l,
        "Follicle No. (R)":     follicle_r,
        "Avg. F size (L) (mm)": fsize_l,
        "Avg. F size (R) (mm)": fsize_r,
        "Weight gain(Y/N)":     weight_gain,
        "hair growth(Y/N)":     hair_growth,
        "Skin darkening (Y/N)": skin_darkening,
        "Pimples(Y/N)":         pimples,
        "Fast food (Y/N)":      fast_food,
        "Reg.Exercise(Y/N)":    exercise,
        "Age (yrs)":            age,
        "Weight (Kg)":          weight,
        "Cycle(R/I)":           cycle_code,
        "Cycle length(days)":   cycle_len_code,
        "TSH (mIU/L)":          tsh,
        "PRL(ng/mL)":           prl,
        "Vit D3 (ng/mL)":       vit_d3,
        "PRG(ng/mL)":           prg,
        "RBS(mg/dl)":           rbs,
        "BP _Systolic (mmHg)":  bp_sys,
        "BP _Diastolic (mmHg)": bp_dia,
        "Endometrium (mm)":     endometrium,
        "Hb(g/dl)":             hemoglobin,
        "Pulse rate(bpm)":      pulse_rate,
    }

    vec = np.array([feature_map[fn] for fn in FEATURE_NAMES], dtype=np.float32)
    return clamp_feature_vector(vec)


def clamp_feature_vector(vec: np.ndarray) -> np.ndarray:
    clamped = vec.copy()
    for i, fname in enumerate(FEATURE_NAMES):
        if fname in _FEATURE_CLAMPS:
            lo, hi = _FEATURE_CLAMPS[fname]
            clamped[i] = float(np.clip(clamped[i], lo, hi))
    return clamped


def build_display_vector(feat_vec: np.ndarray) -> np.ndarray:
    """Return a copy of feat_vec with AMH converted back to ng/mL for display."""
    display_vec = feat_vec.copy()
    display_vec[4] = display_vec[4] / 7.14   # pmol/L → ng/mL
    return display_vec

def compute_clinical_override(data: dict) -> dict:
    flags = []

    bmi = _f(data.get("BMI") or data.get("bmi"), 0)
    if bmi > 30:   flags.append(f"BMI={bmi:.1f} kg/m² (Obese, >30)")
    elif bmi > 27: flags.append(f"BMI={bmi:.1f} kg/m² (Overweight, >27)")

    lh  = _f(data.get("LH"),  0)
    fsh = _f(data.get("FSH"), 1)
    if lh > 15:
        flags.append(f"LH={lh} mIU/mL (elevated, normal <15)")
    if fsh > 0.1 and (lh / fsh) > 2:
        flags.append(f"LH/FSH ratio={lh/fsh:.2f} (>2 is Rotterdam criterion)")

    amh = _f(data.get("AMH"), 0)
    if amh > 3.5:
        flags.append(f"AMH={amh} ng/mL (elevated, normal 1–3.5)")

    cycle = str(data.get("cycle", "R")).strip().upper()
    if cycle == "I":
        flags.append("Menstrual cycle: Irregular (Rotterdam criterion #1)")

    cycle_len = _f(data.get("cycle_length"), 28)
    if cycle_len > 35:
        flags.append(f"Cycle length={cycle_len:.0f} days (oligomenorrhea >35d)")
    elif cycle_len < 21:
        flags.append(f"Cycle length={cycle_len:.0f} days (polymenorrhea <21d)")

    fl = _f(data.get("follicle_L"), 0)
    fr = _f(data.get("follicle_R"), 0)
    if fl > 12 or fr > 12:
        flags.append(
            f"Follicle count L={fl:.0f}/R={fr:.0f} "
            f"(polycystic morphology, >12 per ovary = Rotterdam criterion #3)"
        )

    symptoms = sum([
        _i(data.get("weight_gain",    0)),
        _i(data.get("hair_growth",    0)),
        _i(data.get("skin_darkening", 0)),
        _i(data.get("pimples",        0)),
    ])
    if symptoms >= 3:
        flags.append(f"{symptoms}/4 androgenic symptoms (Rotterdam criterion #2)")
    elif symptoms == 2:
        flags.append(f"{symptoms}/4 androgenic symptoms present")

    rbs = _f(data.get("RBS"), 0)
    if rbs > 140:
        flags.append(f"RBS={rbs} mg/dL (hyperglycaemia, insulin resistance risk)")

    tsh = _f(data.get("TSH"), 0)
    if tsh > 4.0:
        flags.append(f"TSH={tsh} mIU/L (elevated, thyroid involvement)")

    prl = _f(data.get("PRL"), 0)
    if prl > 29:
        flags.append(f"Prolactin={prl} ng/mL (hyperprolactinemia)")

    n = len(flags)
    if n >= 5:
        return {"flags": flags, "count": n,
                "boost": 0.80, "blend_weight": 0.65, "label": "Very High"}
    elif n >= 4:
        return {"flags": flags, "count": n,
                "boost": 0.70, "blend_weight": 0.70, "label": "High"}
    elif n >= 3:
        return {"flags": flags, "count": n,
                "boost": 0.60, "blend_weight": 0.75, "label": "Moderate-High"}
    elif n >= 2:
        return {"flags": flags, "count": n,
                "boost": 0.50, "blend_weight": 0.82, "label": "Moderate"}
    else:
        return {"flags": flags, "count": n,
                "boost": None, "blend_weight": 1.0, "label": "Low"}


def apply_clinical_blend(model_prob: float, override: dict) -> tuple[float, bool]:
    boost        = override.get("boost")
    blend_weight = override.get("blend_weight", 1.0)

    if boost is None:
        return float(model_prob), False

    blended = blend_weight * model_prob + (1.0 - blend_weight) * boost
    if blended > model_prob:
        return float(round(blended, 4)), True
    return float(model_prob), False

def compute_risk_level(probability: float) -> dict:
    if probability >= 0.75:
        return {"level": "High", "color": "var(--rose)",
                "recommendation": (
                    "Multiple strong PCOS indicators detected. Please consult a "
                    "gynaecologist urgently for hormonal panel, pelvic ultrasound, "
                    "and metabolic assessment. Early intervention significantly "
                    "improves long-term outcomes.")}
    elif probability >= 0.5:
        return {"level": "Moderate", "color": "var(--gold)",
                "recommendation": (
                    "Some indicators are elevated. Consult a gynaecologist for a "
                    "thorough evaluation. Consider lifestyle changes including diet "
                    "and exercise. Follow up with hormonal blood tests and an "
                    "ultrasound scan.")}
    elif probability >= 0.35:
        return {"level": "Low-Moderate", "color": "var(--azure)",
                "recommendation": (
                    "A few mild indicators were noted. Maintain a healthy lifestyle "
                    "and schedule a routine gynaecological check-up. Monitor "
                    "menstrual regularity.")}
    else:
        return {"level": "Low", "color": "var(--teal)",
                "recommendation": (
                    "Your indicators are within normal ranges. Maintain a healthy "
                    "diet, regular exercise, and routine gynaecological check-ups "
                    "annually.")}

def get_abnormal_features(feat_vec: np.ndarray) -> list:
    feature_map = dict(zip(FEATURE_NAMES, feat_vec))
    display_map = dict(feature_map)
    if "AMH(ng/mL)" in display_map:
        display_map["AMH(ng/mL)"] = round(display_map["AMH(ng/mL)"] / 7.14, 2)

    flagged = []
    for fname, (lo, hi, unit) in NORMAL_RANGES.items():
        val = display_map.get(fname)
        if val is None:
            continue
        if val < lo:
            flagged.append({"name":   DISPLAY_NAMES.get(fname, fname),
                            "value":  round(float(val), 2),
                            "normal": f"{lo}–{hi} {unit}",
                            "status": "Low"})
        elif val > hi:
            flagged.append({"name":   DISPLAY_NAMES.get(fname, fname),
                            "value":  round(float(val), 2),
                            "normal": f"{lo}–{hi} {unit}",
                            "status": "High"})
    return flagged


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    import cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or corrupted image file")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img_res.astype(np.float32) / 255.0, axis=0)


def validate_input(data: dict) -> dict:
    errors   = []
    required = ["age", "weight", "height", "LH", "FSH", "cycle"]
    for field in required:
        if data.get(field) in [None, ""]:
            errors.append(f"{field} is required")
    return {"valid": len(errors) == 0, "errors": errors}
    bp_sys = _f(data.get('bp_systolic'))
    bp_dia = _f(data.get('bp_diastolic'))
    if bp_sys and bp_dia and bp_sys <= bp_dia:
        errors.append(
            f"Systolic BP ({bp_sys} mmHg) must be greater than "
            f"Diastolic BP ({bp_dia} mmHg)"
        )

    waist = _f(data.get('waist'))
    hip   = _f(data.get('hip'))
    if waist and hip and (waist / hip) > 1.5:
        errors.append(
            f"Waist-to-Hip ratio ({waist / hip:.2f}) is physiologically "
            f"implausible — check waist ({waist} cm) and hip ({hip} cm)"
        )

    lh  = _f(data.get('LH'))
    fsh = _f(data.get('FSH'))
    if lh and fsh and (lh / fsh) > 20:
        errors.append(
            f"LH/FSH ratio ({lh/fsh:.1f}) appears implausible — "
            f"check LH ({lh}) and FSH ({fsh}) values"
        )

    return {'valid': len(errors) == 0, 'errors': errors}


def build_feature_vector(data: dict) -> np.ndarray:
    age    = _f(data.get("age"),    25.0)
    weight = _f(data.get("weight"), 60.0)
    height = _f(data.get("height"), 160.0)
    waist  = _f(data.get("waist"),  32.0)
    hip    = _f(data.get("hip"),    38.0)

    bmi = _f(data.get("BMI"), 0.0)
    if bmi <= 0 and height > 0:
        bmi = weight / ((height / 100.0) ** 2)
    bmi = round(bmi, 2)

    lh  = _f(data.get("LH"),  8.0)
    fsh = _f(data.get("FSH"), 6.0)
    tsh = _f(data.get("TSH"), 2.0)
    prl = _f(data.get("PRL"), 15.0)
    prg = _f(data.get("PRG"), 1.0)

    amh_pmol        = _amh_to_pmol(data.get("AMH", 0.0))
    fsh_lh_ratio    = round(fsh / lh,    4) if lh > 0.1 else 0.0
    waist_hip_ratio = round(waist / hip, 4) if hip > 0   else 0.0

    cycle_code     = _cycle_regularity_code(data.get("cycle", "R"))
    cycle_len_code = _cycle_length_code(data.get("cycle_length", 28))

    follicle_l  = _f(data.get("follicle_L"),  6.0)
    follicle_r  = _f(data.get("follicle_R"),  6.0)
    fsize_l     = _f(data.get("fsize_L"),    10.0)
    fsize_r     = _f(data.get("fsize_R"),    10.0)
    endometrium = _f(data.get("endometrium"),  7.0)
    bp_sys      = _f(data.get("bp_systolic"),  110.0)
    bp_dia      = _f(data.get("bp_diastolic"),  70.0)
    hemoglobin  = _f(data.get("hemoglobin"),   12.0)
    pulse_rate  = _f(data.get("pulse_rate"),   72.0)
    rbs         = _f(data.get("RBS"),          95.0)
    vit_d3      = _f(data.get("vitD3"),        25.0)

    weight_gain    = _i(data.get("weight_gain",    0))
    hair_growth    = _i(data.get("hair_growth",    0))
    skin_darkening = _i(data.get("skin_darkening", 0))
    pimples        = _i(data.get("pimples",        0))
    fast_food      = _i(data.get("fast_food",      0))
    exercise       = _i(data.get("exercise",       0))

    feature_map = {
        "BMI":                   bmi,
        "LH(mIU/mL)":           lh,
        "FSH(mIU/mL)":          fsh,
        "FSH/LH":               fsh_lh_ratio,
        "AMH(ng/mL)":           amh_pmol,
        "Waist:Hip Ratio":      waist_hip_ratio,
        "Follicle No. (L)":     follicle_l,
        "Follicle No. (R)":     follicle_r,
        "Avg. F size (L) (mm)": fsize_l,
        "Avg. F size (R) (mm)": fsize_r,
        "Weight gain(Y/N)":     weight_gain,
        "hair growth(Y/N)":     hair_growth,
        "Skin darkening (Y/N)": skin_darkening,
        "Pimples(Y/N)":         pimples,
        "Fast food (Y/N)":      fast_food,
        "Reg.Exercise(Y/N)":    exercise,
        "Age (yrs)":            age,
        "Weight (Kg)":          weight,
        "Cycle(R/I)":           cycle_code,
        "Cycle length(days)":   cycle_len_code,
        "TSH (mIU/L)":          tsh,
        "PRL(ng/mL)":           prl,
        "Vit D3 (ng/mL)":       vit_d3,
        "PRG(ng/mL)":           prg,
        "RBS(mg/dl)":           rbs,
        "BP _Systolic (mmHg)":  bp_sys,
        "BP _Diastolic (mmHg)": bp_dia,
        "Endometrium (mm)":     endometrium,
        "Hb(g/dl)":             hemoglobin,
        "Pulse rate(bpm)":      pulse_rate,
    }

    vec = np.array([feature_map[fn] for fn in FEATURE_NAMES], dtype=np.float32)
    return clamp_feature_vector(vec)


def clamp_feature_vector(vec: np.ndarray) -> np.ndarray:
    clamped = vec.copy()
    for i, fname in enumerate(FEATURE_NAMES):
        if fname in _FEATURE_CLAMPS:
            lo, hi = _FEATURE_CLAMPS[fname]
            clamped[i] = float(np.clip(clamped[i], lo, hi))
    return clamped


def build_display_vector(feat_vec: np.ndarray) -> np.ndarray:
    display_vec = feat_vec.copy()
    display_vec[4] = display_vec[4] / 7.14
    return display_vec


def compute_clinical_override(data: dict) -> dict:
    flags = []

    bmi = _f(data.get("BMI") or data.get("bmi"), 0)
    if bmi > 30:   flags.append(f"BMI={bmi:.1f} kg/m² (Obese, >30)")
    elif bmi > 27: flags.append(f"BMI={bmi:.1f} kg/m² (Overweight, >27)")

    lh  = _f(data.get("LH"),  0)
    fsh = _f(data.get("FSH"), 1)
    if lh > 15:
        flags.append(f"LH={lh} mIU/mL (elevated, normal <15)")
    if fsh > 0.1 and (lh / fsh) > 2:
        flags.append(f"LH/FSH ratio={lh/fsh:.2f} (>2 is Rotterdam criterion)")

    amh = _f(data.get("AMH"), 0)
    if amh > 3.5:
        flags.append(f"AMH={amh} ng/mL (elevated, normal 1–3.5)")

    cycle = str(data.get("cycle", "R")).strip().upper()
    if cycle == "I":
        flags.append("Menstrual cycle: Irregular (Rotterdam criterion #1)")

    cycle_len = _f(data.get("cycle_length"), 28)
    if cycle_len > 35:
        flags.append(f"Cycle length={cycle_len:.0f} days (oligomenorrhea >35d)")
    elif cycle_len < 21:
        flags.append(f"Cycle length={cycle_len:.0f} days (polymenorrhea <21d)")

    fl = _f(data.get("follicle_L"), 0)
    fr = _f(data.get("follicle_R"), 0)
    if fl > 12 or fr > 12:
        flags.append(
            f"Follicle count L={fl:.0f}/R={fr:.0f} "
            f"(polycystic morphology, >12 per ovary = Rotterdam criterion #3)"
        )

    symptoms = sum([
        _i(data.get("weight_gain",    0)),
        _i(data.get("hair_growth",    0)),
        _i(data.get("skin_darkening", 0)),
        _i(data.get("pimples",        0)),
    ])
    if symptoms >= 3:
        flags.append(f"{symptoms}/4 androgenic symptoms (Rotterdam criterion #2)")
    elif symptoms == 2:
        flags.append(f"{symptoms}/4 androgenic symptoms present")

    rbs = _f(data.get("RBS"), 0)
    if rbs > 140:
        flags.append(f"RBS={rbs} mg/dL (hyperglycaemia, insulin resistance risk)")

    tsh = _f(data.get("TSH"), 0)
    if tsh > 4.0:
        flags.append(f"TSH={tsh} mIU/L (elevated, thyroid involvement)")

    prl = _f(data.get("PRL"), 0)
    if prl > 29:
        flags.append(f"Prolactin={prl} ng/mL (hyperprolactinemia)")

    n = len(flags)
    if n >= 5:
        return {"flags": flags, "count": n,
                "boost": 0.80, "blend_weight": 0.65, "label": "Very High"}
    elif n >= 4:
        return {"flags": flags, "count": n,
                "boost": 0.70, "blend_weight": 0.70, "label": "High"}
    elif n >= 3:
        return {"flags": flags, "count": n,
                "boost": 0.60, "blend_weight": 0.75, "label": "Moderate-High"}
    elif n >= 2:
        return {"flags": flags, "count": n,
                "boost": 0.50, "blend_weight": 0.82, "label": "Moderate"}
    else:
        return {"flags": flags, "count": n,
                "boost": None, "blend_weight": 1.0, "label": "Low"}


def apply_clinical_blend(model_prob: float, override: dict) -> tuple[float, bool]:
    """
    Apply the soft clinical blend and return (final_prob, override_applied).

    Formula:
        final = blend_weight * model_prob + (1 - blend_weight) * boost

    The blend ensures:
        * The ML model always has the majority weight (blend_weight ≥ 0.65)
        * The clinical signal only RAISES the probability (we never lower it)
        * No hard clipping to a fixed value — output varies naturally
    """
    boost        = override.get("boost")
    blend_weight = override.get("blend_weight", 1.0)

    if boost is None:
        return float(model_prob), False

    blended = blend_weight * model_prob + (1.0 - blend_weight) * boost
    if blended > model_prob:
        return float(round(blended, 4)), True
    return float(model_prob), False


def compute_risk_level(probability: float) -> dict:
    if probability >= 0.75:
        return {"level": "High", "color": "var(--rose)",
                "recommendation": (
                    "Multiple strong PCOS indicators detected. Please consult a "
                    "gynaecologist urgently for hormonal panel, pelvic ultrasound, "
                    "and metabolic assessment. Early intervention significantly "
                    "improves long-term outcomes.")}
    elif probability >= 0.5:
        return {"level": "Moderate", "color": "var(--gold)",
                "recommendation": (
                    "Some indicators are elevated. Consult a gynaecologist for a "
                    "thorough evaluation. Consider lifestyle changes including diet "
                    "and exercise. Follow up with hormonal blood tests and an "
                    "ultrasound scan.")}
    elif probability >= 0.35:
        return {"level": "Low-Moderate", "color": "var(--azure)",
                "recommendation": (
                    "A few mild indicators were noted. Maintain a healthy lifestyle "
                    "and schedule a routine gynaecological check-up. Monitor "
                    "menstrual regularity.")}
    else:
        return {"level": "Low", "color": "var(--teal)",
                "recommendation": (
                    "Your indicators are within normal ranges. Maintain a healthy "
                    "diet, regular exercise, and routine gynaecological check-ups "
                    "annually.")}


def get_abnormal_features(feat_vec: np.ndarray) -> list:
    """Flag features outside normal clinical ranges. Converts AMH back to ng/mL."""
    feature_map = dict(zip(FEATURE_NAMES, feat_vec))
    display_map = dict(feature_map)
    if "AMH(ng/mL)" in display_map:
        display_map["AMH(ng/mL)"] = round(display_map["AMH(ng/mL)"] / 7.14, 2)

    flagged = []
    for fname, (lo, hi, unit) in NORMAL_RANGES.items():
        val = display_map.get(fname)
        if val is None:
            continue
        if val < lo:
            flagged.append({"name":   DISPLAY_NAMES.get(fname, fname),
                            "value":  round(float(val), 2),
                            "normal": f"{lo}–{hi} {unit}",
                            "status": "Low"})
        elif val > hi:
            flagged.append({"name":   DISPLAY_NAMES.get(fname, fname),
                            "value":  round(float(val), 2),
                            "normal": f"{lo}–{hi} {unit}",
                            "status": "High"})
    return flagged


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Decode, resize to 224×224 (EfficientNet input), and normalise to [0,1].
    Returns shape (1, 224, 224, 3).
    """
    import cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or corrupted image file")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img_res.astype(np.float32) / 255.0, axis=0)


def validate_input(data: dict) -> dict:
    """
    Basic required-field check. Use validate_medical_input_ranges() for
    full production validation.
    """
    errors   = []
    required = ["age", "weight", "height", "LH", "FSH", "cycle"]
    for field in required:
        if data.get(field) in [None, ""]:
            errors.append(f"{field} is required")
    return {"valid": len(errors) == 0, "errors": errors}