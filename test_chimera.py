##
#Copyright (c) 2026 Mirabela Rusu, Radiology, Stanford University
#This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##

from test_one_case import run_one_case, get_patient_proba
import argparse
import csv
import json
import os
import numpy as np

def get_case_psa(case, base_path):
    """Return the pre-operative PSA for a case, or None if it is not available.

    Looks for a file named "<case>.json" inside the folder given by the
    -c / --clinica_var_psa_json_path option and reads its "preoperative_PSA" key.
    Returns None when the option was not set, the file is missing, or the key
    is absent.
    """
    # No clinical-variable folder was provided on the command line.
    if not args.clinica_var_psa_json_path:
        return None
    # One json file per case, named after the case id.
    psa_json_path = os.path.join(base_path, case + ".json")
    if not os.path.isfile(psa_json_path):
        print("Couldn't find PSA json for case", case, ":[", psa_json_path, "]")
        return None
    # .get() yields None if the "preoperative_PSA" key is missing.
    with open(psa_json_path, "r") as f:
        return json.load(f).get("pre_operative_PSA")
        
if __name__=="__main__":
    print("Get predictions for one case")
    parser = argparse.ArgumentParser(description='Run inference for all models for the chimera data')
    parser.add_argument('--input', '-i', type=str, 
                        required=False,
                        default='./path_to_chimera_data/',
                        help='path to data, each patient, one folder, with key words, t2 and adc in filename')
    parser.add_argument('--output', '-o', type=str,
                        default='./results/chimera',
                        required=False, 
                        help='folder where to put all the results')
    parser.add_argument('--proba_threshold', '-p', type=float,
                        default=0.1,
                        required=False, 
                        help='what probability to use to threshold the output')
    parser.add_argument('--process_all', '-a', action='store_true',
                        default=False,
                        required=False,
                        help='should all cases be processed? Default: False')
    parser.add_argument('--clinica_var_psa_json_path', '-c', type=str,
                        default=None,
                        required=False,
                        help='path to a folder with one json file per case, each containing a "preoperative_PSA" key')
        
    args = parser.parse_args()


    ######
    ## trained nnUnet Models
    ## key -> path of the nnUNet model folder used for that prediction task
    ######
    model_paths = {'csPCA':"models/Dataset202_BxMR_withRegions_T2_ADC/nnUNetTrainer__nnUNetPlans__3d_fullres/",
                       'aggInd':"models/Dataset203_CaAggInd_i4ch_oIndAggCh_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/",
                       'KI67':"models/Dataset361_12342_MKI67_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres",
                       'Metastasis':"models/Dataset309_Decipher_som_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
                      }
    ### check that every model folder is present before doing any work
    if not os.path.exists('models'):
        print("Can't find the models. Please create the folder 'models', to includes the trained models")
        exit()

    for p in model_paths.keys():
        if not os.path.exists(model_paths[p]):
            print("Cant find a model folder", model_paths[p])
            exit ()

    # Input folder must exist; it holds one sub-folder per patient/case.
    if not os.path.exists(args.input):
        print("Path doesn't exist", args.input)
        exit()

    all_items = os.listdir(args.input)

    # Keep only the sub-directories (the cases) and sort them for a stable order.
    cases = np.sort([f for f in all_items if os.path.isdir(os.path.join(args.input, f ))])

    # PSA value per case, aligned index-for-index with `cases` (None when unknown).
    psa_values = [get_case_psa(case, args.clinica_var_psa_json_path) for case in cases]

    # Open the CSV that collects per-case stats; header is written lazily once
    # we see the first row so the columns can come from run_one_case's output.
    # If stats.csv already exists (and is non-empty) open it in append mode and
    # keep the existing header instead of overwriting the file.
    csv_path = os.path.join(args.output, "stats.csv")
    os.makedirs(args.output, exist_ok=True)
    append_mode = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, "a" if append_mode else "w", newline="")
    writer = None

    for idx, case in enumerate(cases):
        # Unless -a/--process_all is set, stop after the first 3 cases (quick test run).
        if not args.process_all and idx > 2:
            print("Done processing 3, exiting now as option to process all is false. \n",
                  "if you want to process all cases, then use flag -a or --process_all.")
            break
        print("**** Processing", idx, " - id: ", case)
        # Skip cases that already have an output folder (resume a partial run).
        outpath = os.path.join(args.output, case)
        if os.path.exists(outpath):
            print("Skipping processed case", case)
            continue

        case_path = os.path.join(args.input, case)
        print(case_path)

        # Pre-operative PSA for this case (may be None).
        psa = psa_values[idx]

        # Look at the files in the case folder and pick out the T2 and ADC volumes
        # by matching "t2" / "adc" anywhere in the (lower-cased) filename.
        all_file_items = os.listdir (case_path)
        files = np.sort([f for f in all_file_items if os.path.isfile(os.path.join(case_path,f))])
        t2_path = ""
        adc_path = ""
        for f in files:
            case_id = f[:len(f)-6]  # strip the trailing extension (e.g. ".nii.gz")
            print(f, case_id)
            if "t2" in f.lower(): # found t2
                t2_path = os.path.join(case_path, f)
            if "adc" in f.lower(): # found adc
                adc_path = os.path.join(case_path, f)
        outpath = os.path.join(args.output, case)

        # Only run inference when both required modalities were found.
        if len(t2_path)>0 and len(adc_path)>0:
            stats = run_one_case(t2_path, adc_path, outpath, model_paths,case,args.proba_threshold)

            th_epe = 0.4873275104221620
            feats = [[stats['pro'], psa, stats['cln']]]
            epe_model_path = os.path.join('models/patient_level/rf2_epe.pkl')        
            epe_scalar_path = os.path.join('models/patient_level/scaler_epe.pkl')        
            epe_proba = get_patient_proba(epe_scalar_path, epe_model_path, feats)    
        
            feats = [[psa, stats['hi3'], stats['prf']]]
            # compute bcr probability
            th_bcr = 0.10545787330199500
            brc_model_path = os.path.join('models/patient_level/rf2_bcr.pkl')        
            bcr_scalar_path = os.path.join('models/patient_level/scaler_bcr.pkl')        
            bcr_proba = get_patient_proba(bcr_scalar_path, brc_model_path, feats)
        
            print("Computed EPE probability:", epe_proba[0], "-> Binary EPE Status: ", float(epe_proba[0])>th_epe, 
                    "\nComputed BRC probability:", bcr_proba[0] ,"-> Binary BRC Status: ", float(bcr_proba[0])>th_bcr)    

            
            if stats:
                # Patient-level predictions, their decision thresholds, and the
                # resulting binary status (probability > threshold).
                stats["epe_proba"] = float(epe_proba[0])
                stats["epe_threshold"] = th_epe
                stats["epe_status"] = float(epe_proba[0]) > th_epe
                stats["bcr_proba"] = float(bcr_proba[0])
                stats["bcr_threshold"] = th_bcr
                stats["bcr_status"] = float(bcr_proba[0]) > th_bcr

                # Prepend the case id, then append this case's stats as a CSV row.
                row = {"case_id": case, **stats}
                if writer is None:
                    # First row: create the writer, and emit the header only when
                    # starting a fresh file (not when appending to an existing one).
                    writer = csv.DictWriter(csv_file, fieldnames=row.keys())
                    if not append_mode:
                        writer.writeheader()
                writer.writerow(row)
                csv_file.flush()  # flush so partial results survive a crash
        else:
            print("Issues with the path. \nT2:[",t2_path,"]\nADC:[",adc_path,"].",len(t2_path), len(adc_path))

    csv_file.close()
    print("Stats written to", csv_path)

