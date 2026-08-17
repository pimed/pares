##
#Copyright (c) 2026 Mirabela Rusu, Radiology, Stanford University
#This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##

from test_one_case import run_one_case
import argparse
import csv
import os
import numpy as np

if __name__=="__main__":
    print("Get predictions for one case")
    parser = argparse.ArgumentParser(description='Run inference for all models for the chimera data')
    parser.add_argument('--input', '-i', type=str, 
                        required=False,
                        default='./data/2025_Chimera/images/',
                        help='path to data, each patient, one folder, with key words, t2 and adc in filename')
    parser.add_argument('--output', '-o', type=str,
                        default='./data/gene_inference/',
                        required=False, 
                        help='folder where to put all the results')
    parser.add_argument('--proba_threshold', '-p', type=float,
                        default=0.5,
                        required=False, 
                        help='what probability to use to threshold the output')
    args = parser.parse_args()


    ######
    ## trained nnUnet Models 
    ######
    model_paths = {'csPCA':"models/Dataset202_BxMR_withRegions_T2_ADC/nnUNetTrainer__nnUNetPlans__3d_fullres/",
                  'aggInd':"models/Dataset203_CaAggInd_i4ch_oIndAggCh_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/",
                  'KI67':"models/Dataset361_12342_MKI67_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres",
                  'Metastasis':"models/Dataset309_Decipher_som_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
                  }
    ### check if they exist
    if not os.path.exists('models'):
        print("Can't find the models. Please create the folder 'models', to includes the trained models")
        exit()
    
    for p in model_paths.keys():
        if not os.path.exists(model_paths[p]):
            print("Cant find a model folder", model_paths[p])
            exit ()

    if not os.path.exists(args.input):
        print("Path doesn't exist", args.input)
        exit()

    all_items = os.listdir(args.input)

    cases = np.sort([f for f in all_items if os.path.isdir(os.path.join(args.input, f ))])

    csv_path = os.path.join(args.output, "stats.csv")
    os.makedirs(args.output, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = None

    for case in cases:
        print("**** Processing", case)
        outpath = os.path.join(args.output, case)
        if os.path.exists(outpath):
            print("Skipping processed case", case)
            continue

        case_path = os.path.join(args.input, case)
        print(case_path)
        all_file_items = os.listdir (case_path)
        files = np.sort([f for f in all_file_items if os.path.isfile(os.path.join(case_path,f))])
        t2_path = ""
        adc_path = ""
        for f in files: 
            case_id = f[:len(f)-6]
            print(f, case_id)
            if "t2" in f.lower(): # found t2
                t2_path = os.path.join(case_path, f)
            if "adc" in f.lower(): # found adc
                adc_path = os.path.join(case_path, f)
        outpath = os.path.join(args.output, case)

        if len(t2_path)>0 and len(adc_path)>0:
            stats = run_one_case(t2_path, adc_path, outpath, model_paths,case,args.proba_threshold)
            if stats:
                row = {"case_id": case, **stats}
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
        else:
            print("Issues with the path. \nT2:[",t2_path,"]\nADC:[",adc_path,"].",len(t2_path), len(adc_path))

    csv_file.close()
    print("Stats written to", csv_path)

