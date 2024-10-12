
def output_segment_desc(desc_dict: dict, wfile):
    print(f"subject: {desc_dict['SUBJECT_ID']}, ", end='', file=wfile)
    print(f"icustay: {desc_dict['ICUSTAY_ID']}, ", end='', file=wfile)
    if 'HCV ErrorMsg' in desc_dict:
        print("\n\tHCV", file=wfile)
        print(f"\t\tErrorMsg: {desc_dict['HCV ErrorMsg']}", end='', file=wfile)
    else:
        print(f"mortality: {desc_dict['MORTALITY_INUNIT']}, ",
              end='', file=wfile)
        print(f"base_los2: {desc_dict['seg_base_los2']:.1f}, ",
              end='', file=wfile)
        print(f"end_los2: {desc_dict['seg_end_los2']:.1f}, ",
              end='', file=wfile)
        print(f"los2: {desc_dict['LOS2']:.1f}", file=wfile)
        if 'Time_Window' in desc_dict:
            print("\tTime_Window", file=wfile)
            for key, val in desc_dict['Time_Window'].items():
                print(f"\t\t{key}: {val}", file=wfile)

        print("\tHCV", file=wfile)
        print(f"\t\tflag: {desc_dict['hcv_flag']}", file=wfile)
        if desc_dict['hcv_desc']:
            for key, val in desc_dict['hcv_desc'].items():
                print(f"\t\t{key}: {val}", file=wfile)
        else:
            print(f"\t\t{desc_dict['hcv_desc']}", file=wfile)

        print("\tNUME", file=wfile)
        print(f"\t\tflag: {desc_dict['nume_flag']}", file=wfile)
        if desc_dict['nume_desc']:
            for key, val in desc_dict['nume_desc'].items():
                print(f"\t\t{key}: {val}", file=wfile)
        else:
            print(f"\t\t{desc_dict['nume_desc']}", file=wfile)

        print("\tWAVE", file=wfile)
        print(f"\t\tflag: {desc_dict['wave_flag']}", file=wfile)
        if desc_dict['wave_desc']:
            for key, val in desc_dict['wave_desc'].items():
                print(f"\t\t{key}: {val}", file=wfile)
        else:
            print(f"\t\t{desc_dict['wave_desc']}", file=wfile)

    print('', file=wfile, flush=True)
