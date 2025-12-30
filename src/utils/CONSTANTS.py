LABEL_COLS: list[str] = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]
CAT_COLS = LABEL_COLS[:-1]  # all except 'Aneurysm Present'
MODALITIES = ["MRA", "CTA", "MRI T2", "MRI T1post"]
MODALITY_TO_INT = {mod: i for i, mod in enumerate(MODALITIES)}

# Training parameters pre-computed on dataset
POS_WEIGHT = 5.75  # for handling class imbalance in aneurysm present // n_neg / n_pos
CLASS_WEIGHTS = [1.0] * len(
    CAT_COLS
)  # for location classes // dummy value, to be replaced with class weights in dataset
