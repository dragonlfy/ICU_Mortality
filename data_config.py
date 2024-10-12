from numpy import NaN

mimic3csvs_dir = 'data/mimic3csvs'  # raw files of Tables
# stay recordings obtained by breaking up tables by subjects and stays
mimic3stays_dir = 'data/mimic3stays'
icustay_desc_path = 'data/icustay_desc.csv'
# icustay description
var_est_icustay_desc_path = 'data/var_est_icustay_desc.csv'
# icustay description
variable_map_path = 'datautils/itemid_to_variable_map.csv'
phenotype_definitions_path = 'datautils/hcup_ccs_2015_definitions.yaml'
matched_path = 'data/mimic3waveform/mimic3wdb-matched'
rpath = 'data/mimic3waveform/mimic3wdb-matched/RECORDS'
npath = 'data/mimic3waveform/mimic3wdb-matched/RECORDS-numerics'
wpath = 'data/mimic3waveform/mimic3wdb-matched/RECORDS-waveforms'
wn_recs_path = 'data/wn_recs.csv'
wn_rec_feat_path = 'data/wn_rec_features.csv'
recording_missing_path = 'data/recording_missing.txt'
wnhcv_desc_path = 'data/wnhcv_desc.csv'
phenotype_label_path = 'data/phenotype_labels.csv'


HCV_Var_Converters = {
    "Capillary refill rate":
        (lambda val: {0.0: 0, 1.0: 1}.get(val, NaN)),
    "Diastolic blood pressure":
        (lambda val: NaN if val == "" else float(val)),
    "Fraction inspired oxygen":
        (lambda val: NaN if val == "" else float(val)),
    "Glascow coma scale eye opening":
        (lambda val: {
            "None": 0,
            "1 No Response": 1,
            "2 To pain": 2,
            "To Pain": 2,
            "3 To speech": 3,
            "To Speech": 3,
            "4 Spontaneously": 4,
            "Spontaneously": 4,
        }.get(val, NaN)),
    "Glascow coma scale motor response":
        (lambda val: {
            "1 No Response": 1-1,
            "No response": 1-1,
            "2 Abnorm extensn": 2-1,
            "Abnormal extension": 2-1,
            "3 Abnorm flexion": 3-1,
            "Abnormal Flexion": 3-1,
            "4 Flex-withdraws": 4-1,
            "Flex-withdraws": 4-1,
            "5 Localizes Pain": 5-1,
            "Localizes Pain": 5-1,
            "6 Obeys Commands": 6-1,
            "Obeys Commands": 6-1
        }.get(val, NaN)),
    "Glascow coma scale total":
        (lambda val: {
            "3": 3-3,
            "4": 4-3,
            "5": 5-3,
            "6": 6-3,
            "7": 7-3,
            "8": 8-3,
            "9": 9-3,
            "10": 10-3,
            "11": 11-3,
            "12": 12-3,
            "13": 13-3,
            "14": 14-3,
            "15": 15-3,
        }.get(val, NaN)),
    "Glascow coma scale verbal response":
        (lambda val: {
            "No Response-ETT": 1-1,
            "No Response": 1-1,
            "1 No Response": 1-1,
            "1.0 ET/Trach": 1-1,
            "2 Incomp sounds": 2-1,
            "Incomprehensible sounds": 2 - 1,
            "3 Inapprop words": 3-1,
            "Inappropriate Words": 3-1,
            "4 Confused": 4-1,
            "Confused": 4-1,
            "5 Oriented": 5-1,
            "Oriented": 5-1
        }.get(val, NaN)),
    "Glucose":
        (lambda val: NaN if val == "" else float(val)),
    "Heart Rate":
        (lambda val: NaN if val == "" else float(val)),
    "Height":
        (lambda val: NaN if val == "" else float(val)),
    "Mean blood pressure":
        (lambda val: NaN if val == "" else float(val)),
    "Oxygen saturation":
        (lambda val: NaN if val == "" else float(val)),
    "Respiratory rate":
        (lambda val: NaN if val == "" else float(val)),
    "Systolic blood pressure":
        (lambda val: NaN if val == "" else float(val)),
    "Temperature":
        (lambda val: NaN if val == "" else float(val)),
    "Weight":
        (lambda val: NaN if val == "" else float(val)),
    "pH":
        (lambda val: NaN if val == "" else float(val)),
}


HCV_Is_Categorical_Channel = {
    'Capillary refill rate': True,
    'Diastolic blood pressure': False,
    'Fraction inspired oxygen': False,
    'Glascow coma scale eye opening': True,
    'Glascow coma scale motor response': True,
    'Glascow coma scale total': True,
    'Glascow coma scale verbal response': True,
    'Glucose': False,
    'Heart Rate': False,
    'Height': False,
    'Mean blood pressure': False,
    'Oxygen saturation': False,
    'Respiratory rate': False,
    'Systolic blood pressure': False,
    'Temperature': False,
    'Weight': False,
    'pH': False,
}

HCV_Var_Impute_Value = {
    'Capillary refill rate': 0,
    'Diastolic blood pressure': 59,
    'Fraction inspired oxygen': 0.21,
    'Glascow coma scale eye opening': 4,
    'Glascow coma scale motor response': 5,
    'Glascow coma scale total': 12,
    'Glascow coma scale verbal response': 4,
    'Glucose': 128,
    'Heart Rate': 86,
    'Height': 170,
    'Mean blood pressure': 77,
    'Oxygen saturation': 98,
    'Respiratory rate': 19,
    'Systolic blood pressure': 118,
    'Temperature': 36,
    'Weight': 81,
    'pH': 7.4,
}

HCV_Num_Possible_Values = {
    'Capillary refill rate': 2,
    'Glascow coma scale eye opening': 5,
    'Glascow coma scale motor response': 6,
    'Glascow coma scale total': 13,
    'Glascow coma scale verbal response': 5,
}
