##
#Copyright (c) 2026 Mirabela Rusu, Radiology, Stanford University
#This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
#To view a copy of this license, visit http://creativecommons.org or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
##
import os
import SimpleITK as sitk
import argparse
import numpy as np
import torch
import pickle

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
DEBUG = False

def run_nnUnet_inference(image_data, data_properties, model_path, folds=(0,)):
    """image_data - concatenated np.array?"""
    # image_data: Shape must be (C, X, Y, Z) for 3D or (C, X, Y) for 2D. dtype should be float32
    # data_properties: A dictionary containing spacing and other metadata
    try: 
        # 1. Initialize the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            #perform_everything_on_device=False, # Set to True for max speed if you have enough VRAM
            #device=torch.device('cpu'), 
            perform_everything_on_device=True, # Set to True for max speed if you have enough VRAM
            #device=torch.device('0'), 
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # 2. Load your model weights
        # Point this to your nnUNet_results directory where the model was trained
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_path,
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth"
        )
   
        # 4. Predict a single numpy array
        predicted_segmentation, proba = predictor.predict_single_npy_array(
            image_data, 
            data_properties, 
            segmentation_previous_stage=None, 
            #output_file_or_directory=None, # Set to None to return segmentation array in-memory
            save_or_return_probabilities=True,
        )
    except Exception as e:
        try:
            print('Couldn\'t run! nnUnet error from predict \"', e,"\".")
            print('**** Trying to run on macs with Mx processor. *****')
            # 1. Initialize the predictor
            predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=True,
                    perform_everything_on_device=False, # Set to True for max speed if you have enough VRAM
                    device=torch.device('mps'), 
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=True
                )
        
            # 2. Load your model weights
            # Point this to your nnUNet_results directory where the model was trained
            predictor.initialize_from_trained_model_folder(
                    model_training_output_dir=model_path,
                    use_folds=folds,
                    checkpoint_name="checkpoint_final.pth"
                )

            # 4. Predict a single numpy array
            predicted_segmentation, proba = predictor.predict_single_npy_array(
                image_data, 
                data_properties, 
                segmentation_previous_stage=None, 
                #output_file_or_directory=None, # Set to None to return segmentation array in-memory
                save_or_return_probabilities=True,
            )
        except Exception as eee:

            try:
                print('Couldn\'t run! nnUnet error from predict \"', eee,"\".")
                print('**** Trying to run on CPU. It might work, but it will be sloooooowwww. *****')
                # 1. Initialize the predictor
                predictor = nnUNetPredictor(
                        tile_step_size=0.5,
                        use_gaussian=True,
                        use_mirroring=True,
                        perform_everything_on_device=True, # Set to True for max speed if you have enough VRAM
                        device=torch.device('cpu'), 
                        verbose=False,
                        verbose_preprocessing=False,
                        allow_tqdm=True
                    )
            
                # 2. Load your model weights
                # Point this to your nnUNet_results directory where the model was trained
                predictor.initialize_from_trained_model_folder(
                        model_training_output_dir=model_path,
                        use_folds=folds,
                        checkpoint_name="checkpoint_final.pth"
                    )
                
                # 4. Predict a single numpy array
                predicted_segmentation, proba = predictor.predict_single_npy_array(
                    image_data, 
                    data_properties, 
                    segmentation_previous_stage=None, 
                    #output_file_or_directory=None, # Set to None to return segmentation array in-memory
                    save_or_return_probabilities=True,
                )
            except Exception as ee:
                print("nnUnet error", ee)
                print("ERROR: both gpu and cpu run failed. Its probably a library problem. \n",
                    "Check your pytorch and nnUnet libraries!")
                return (None, None)

    print("Segmentation shape:", predicted_segmentation.shape)
    print("Unique classes:", np.unique(predicted_segmentation))
    #print("Unique classes:", proba)
    

    return (predicted_segmentation, proba)

def get_patient_proba(scaler_path, model_path, feats):
    """loads and runs the models with the three parameters  """            
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Couldn't find", scalar_path, " or ", model_path, 
                ". EPE and BcR can't be computed")
        return -1, -1 
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print("Succesfully opened random forest model.")

    X       = scaler.transform(feats)
    y_score = model.predict_proba(X)[:, 1]
        
    return y_score

def compute_label_volume(image, label):
    """Returns the volume in mm3 of voxels matching `label` (int or list of ints) in a SimpleITK image."""
    arr = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    labels = [label] if np.isscalar(label) else label
    return int(np.sum(np.isin(arr, labels))) * voxel_vol_mm3


def filter_components_by_overlap(image1, image2, dice_threshold=0.10):
    """
    Returns a copy of image1 with connected components removed if they do not
    overlap with any connected component in image2 at Dice >= dice_threshold.

    Both inputs are treated as binary (non-zero = foreground).
    """
    bin1 = sitk.Cast(image1 != 0, sitk.sitkUInt8)
    bin2 = sitk.Cast(image2 != 0, sitk.sitkUInt8)

    cc1 = sitk.ConnectedComponent(bin1)
    cc2 = sitk.ConnectedComponent(bin2)

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(cc1)
    labels1 = shape_filter.GetLabels()
    sizes1 = {lbl: shape_filter.GetNumberOfPixels(lbl) for lbl in labels1}

    shape_filter.Execute(cc2)
    labels2 = shape_filter.GetLabels()
    sizes2 = {lbl: shape_filter.GetNumberOfPixels(lbl) for lbl in labels2}

    # Count all pairwise voxel intersections in one pass — no per-component masks.
    arr1 = sitk.GetArrayViewFromImage(cc1).ravel().astype(np.int64)
    arr2 = sitk.GetArrayViewFromImage(cc2).ravel().astype(np.int64)
    n2 = int(arr2.max()) + 1 if arr2.size else 1
    pair_counts = np.bincount(arr1 * n2 + arr2)

    labels_to_remove = []
    for lbl1 in labels1:
        max_dice = 0.0
        for lbl2 in labels2:
            idx = lbl1 * n2 + lbl2
            intersection = int(pair_counts[idx]) if idx < len(pair_counts) else 0
            if intersection == 0:
                continue
            dice = 2.0 * intersection / (sizes1[lbl1] + sizes2[lbl2])
            if dice > max_dice:
                max_dice = dice
            if max_dice >= dice_threshold:
                break

        if max_dice < dice_threshold:
            labels_to_remove.append(lbl1)

    if labels_to_remove:
        change_filter = sitk.ChangeLabelImageFilter()
        change_filter.SetChangeMap({int(lbl): 0 for lbl in labels_to_remove})
        cc1 = change_filter.Execute(cc1)

    result = sitk.Cast(cc1 != 0, sitk.sitkUInt8)
    result.CopyInformation(image1)
    return result



def run_one_case(t2_path, adc_path, out_path, model_paths, case_id, proba_threshold=0.5, filter_by_csPCA = True):
    """Takes as input a T2 and ADC, creates the rest of the outputs
       uses proba_threshold to set what proba to be consider the label, default 0.5
    """

    if os.path.exists(out_path):
        print("Folder ", out_path, " exits. \n " )
        over_write = input("Do you want to overwrite its content Y/[N]?")
        if over_write in ['y','Y', 'Yes','yes']:
            print("Over Writing the content of ", out_path)
        else:
            print("Exiting now!")
            return
    if not os.path.exists(t2_path):
        print("No file found in ", t2_path,"\nT2 needed. Use option --t2 to set the path to the T2 file")
        return()

    try: 
        t2 = sitk.ReadImage(t2_path)
    except Exception as e:
        print("Something went wrong when reading T2 image! We cant proceed!! Exiting...\n")
        print(e)
        return()

    try: 
        adc = sitk.ReadImage(adc_path)
    except Exception as e:
        print("Something went wrong when reading ADC! We cant proceed!!\nCheck your option --adc.\n")
        print(e)
        return()

    print("**** Writing results to", out_path)
    os.makedirs(out_path,exist_ok = True )
    
    """
    Prepare the data for nnUnet run through API not console
    """
    t2_arr = sitk.GetArrayFromImage(t2)

    #resample first on t2 so they are in the same space then get the adc array
    adc_arr = sitk.GetArrayFromImage(sitk.Resample(adc,t2, sitk.Transform()))

    #create the stacked input data for nnunet
    im_data = np.zeros((2, t2_arr.shape[0],t2_arr.shape[1],t2_arr.shape[2]))
    im_data[0,:,:,:] = t2_arr
    im_data[1,:,:,:] = adc_arr

    data_properties = {
        'spacing': [t2.GetSpacing()[2],t2.GetSpacing()[0],t2.GetSpacing()[1]],
        'orig_spacing': t2.GetOrigin(),
        # other nnU-Net required metadata props
    }

    
    ######
    ### Get the csPCA from MRI. 
    #####
    #Use only fold 0
    seg_arr, proba = run_nnUnet_inference(im_data, data_properties, model_paths['csPCA'], (0,))
    # channel 0 is prostate, 1 is cancer, 2 is csPCA 

    # no segmentation arr was created, likely due to nnUnet/library issues
    if seg_arr is None : 
        print('Something whent wrong with nnUnet run! Exiting now.')
        exit()
    
    pr_proba = proba[0] 
    csPCa_proba = proba[2] 

    seg = sitk.GetImageFromArray(seg_arr)
    seg.CopyInformation(t2)

    fn = os.path.join(out_path, case_id+"_csp_label.nii.gz")
    sitk.WriteImage(seg, fn)
    print("  Done writing clinically significant cancer label (th: " + str(proba_threshold) + "):", fn)

    # prostate proba
    pr_proba_im = sitk.GetImageFromArray(pr_proba)
    pr_proba_im.CopyInformation(t2)
    
    fn = os.path.join(out_path, case_id+"_pro_proba.nii.gz")
    sitk.WriteImage(pr_proba_im, fn)
    print("  Done writing the prostate proba (th: " + str(proba_threshold) + "):", fn)


    csPCa_proba_im = sitk.GetImageFromArray(csPCa_proba)
    csPCa_proba_im.CopyInformation(t2)

    fn = os.path.join(out_path, case_id+ "_csp_proba.nii.gz")
    sitk.WriteImage(csPCa_proba_im, fn)
    print("  Done writing clinically significant cancer proba (th: " + str(proba_threshold) + "):", fn)

    ######
    ### Get agg vs indolent from MRI
    #####
    im_data = np.zeros((4, t2_arr.shape[0],t2_arr.shape[1],t2_arr.shape[2]))
    im_data[0,:,:,:] = t2_arr
    im_data[1,:,:,:] = adc_arr
    im_data[2,:,:,:] = csPCa_proba
    im_data[3,:,:,:] = pr_proba


    seg_aggVsInd_arr_both, proba = run_nnUnet_inference(im_data, data_properties, model_paths['aggInd'],(0,1,2,3,4,))
    
    #both agg and Indolent
    seg_aggVsInd_both = sitk.GetImageFromArray(seg_aggVsInd_arr_both)
    seg_aggVsInd_both.CopyInformation(t2)

    fn = os.path.join(out_path, case_id+"_agi_label.nii.gz")
    sitk.WriteImage(seg_aggVsInd_both, fn)
    print("  Done writing aggressive and indolent lables by Grade group (1 - indolent, 2 - aggressive):", fn)
    

    #just aggressive
    gene_proba_agg = proba[2]
    seg_aggVsInd_arr = (gene_proba_agg>=proba_threshold).astype('uint8') # 0-medium, 1-high  
    seg_aggVsInd = sitk.GetImageFromArray(seg_aggVsInd_arr)
    seg_aggVsInd.CopyInformation(t2)

    if DEBUG: 
        fn = os.path.join(out_path, case_id+"_agg_label.nii.gz")
        sitk.WriteImage(seg_aggVsInd, fn)
        print("  Done writing aggressive lables (value 1 indicates GG>=2):", fn)
    
    gene_proba_im = sitk.GetImageFromArray(gene_proba_agg)
    gene_proba_im.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id+ "_agg_proba.nii.gz")
        sitk.WriteImage(gene_proba_im, fn)
        print("  Done writing aggressive probability:", fn)
    
    # postprocessing step to reduce false positives, by removing predicted regions that 
    # don't overlap with aggressive predictions at least at 1 % threshold
    if filter_by_csPCA:
        seg_aggVsInd_filtered = filter_components_by_overlap(seg_aggVsInd,sitk.Cast(seg>1,sitk.sitkUInt8),0.01)
        fn = os.path.join(out_path, case_id+ "_agg_f_label.nii.gz")
        sitk.WriteImage(seg_aggVsInd_filtered, fn)
        print("  Done writing filtered aggressive labels:", fn)

        if DEBUG:
            seg_aggVsInd_filtered_proba = gene_proba_im*sitk.Cast(seg_aggVsInd_filtered >0,sitk.sitkFloat32)
            fn = os.path.join(out_path, case_id+ "_agg_f_proba.nii.gz")
            sitk.WriteImage(seg_aggVsInd_filtered_proba, fn)
            print("  Done writing filtered aggressive probability:", fn)

    ######
    ### Get ki67 - proliferation from MRI
    #####
    seg_ki67_arr, proba = run_nnUnet_inference(im_data, data_properties, model_paths['KI67'],(0,1,2,3,4,))
    
    gene_proba_ki = proba[2]
    seg_ki67_arr = (gene_proba_ki>proba_threshold).astype('uint8')
    seg_ki67 = sitk.GetImageFromArray(seg_ki67_arr)
    seg_ki67.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id+ "_prf_label.nii.gz")
        sitk.WriteImage(seg_ki67, fn)
        print("  Done writing proliferation labels:", fn)

    gene_proba_im = sitk.GetImageFromArray(gene_proba_ki)
    gene_proba_im.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id+ "_prf_proba.nii.gz")
        sitk.WriteImage(gene_proba_im, fn)
        print("  Done writing proliferation probability:", fn)

    # postprocessing step to reduce false positives, by removing predicted regions that 
    # don't overlap with aggressive predictions at least at 1 % threshold
    if filter_by_csPCA:
        seg_ki67_filtered = filter_components_by_overlap(seg_ki67,sitk.Cast(seg>1,sitk.sitkUInt8),0.01)
        fn = os.path.join(out_path, case_id+ "_prf_f_label.nii.gz")
        sitk.WriteImage(seg_ki67_filtered, fn)
        print("  Done writing filtered proliferation labels:", fn)

        if DEBUG:
            seg_ki67_filtered_proba = gene_proba_im*sitk.Cast(seg_ki67_filtered>0,sitk.sitkFloat32)
            fn = os.path.join(out_path, case_id+ "_prf_f_proba.nii.gz")
            sitk.WriteImage(seg_ki67_filtered_proba, fn)
            print("  Done writing filtered proliferation probability:", fn)

    ######
    ### Get Metastasis from MRI
    #####
    print("Compute the region of high metastasis likelihood")
    seg_de_arr, proba = run_nnUnet_inference(im_data, data_properties, model_paths['Metastasis'],(0,1,2,3,4,))
    
    gene_proba_de = proba[2]
    seg_de_arr = (gene_proba_de>proba_threshold).astype('uint8')
    seg_de = sitk.GetImageFromArray(seg_de_arr)
    seg_de.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id+ "_met_label.nii.gz")
        sitk.WriteImage(seg_de, fn)

    gene_proba_im = sitk.GetImageFromArray(gene_proba_de)
    gene_proba_im.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id+ "_met_proba.nii.gz")
        sitk.WriteImage(gene_proba_im, fn)
        print("  Done writing metastasis risk probability:", fn)

    # postprocessing step to reduce false positives, by removing predicted regions that 
    # don't overlap with aggressive predictions at least at 1 % threshold
 
    if filter_by_csPCA:
        seg_de_filtered = filter_components_by_overlap(seg_de,sitk.Cast(seg>1,sitk.sitkUInt8),0.01)
        fn = os.path.join(out_path, case_id+ "_met_f_label.nii.gz")
        sitk.WriteImage(seg_de_filtered, fn)
        print("  Done writing filtered metastasis risk label:", fn)

        if DEBUG:
            seg_de_filtered_proba = gene_proba_im*sitk.Cast(seg_de_filtered>0,sitk.sitkFloat32)
            fn = os.path.join(out_path, case_id+ "_met_f_proba.nii.gz")
            sitk.WriteImage(seg_de_filtered_proba, fn)
            print("  Done writing filtered metastasis risk probability:", fn)

    ######
    ### Combined 3-channel vector image: Metastasis, prolif, AggVsInd
    #####
    combined_arr = np.stack([seg_aggVsInd_arr, seg_ki67_arr, seg_de_arr], axis=-1).astype('uint8')
    vector_image = sitk.GetImageFromArray(combined_arr, isVector=True)
    vector_image.CopyInformation(t2)

    if DEBUG:
        fn = os.path.join(out_path, case_id + "_mpa.nii.gz")
        sitk.WriteImage(vector_image, fn)
        print("  Done writing combined 3 biomarkers as verctor file:", fn)

    if filter_by_csPCA:
        combined_arr = np.stack([sitk.GetArrayFromImage(seg_aggVsInd_filtered), 
            sitk.GetArrayFromImage(seg_ki67_filtered), 
            sitk.GetArrayFromImage(seg_de_filtered)], axis=-1).astype('uint8')
        vector_image = sitk.GetImageFromArray(combined_arr, isVector=True)
        vector_image.CopyInformation(t2)

        fn = os.path.join(out_path, case_id + "_mpa_f.nii.gz")
        sitk.WriteImage(vector_image, fn)
        print("  Done writing combined 3 biomarkers as verctor file:", fn)


    combinedIm= sitk.GetArrayFromImage(seg_aggVsInd_filtered)+ \
        sitk.GetArrayFromImage(seg_ki67_filtered)+ \
        sitk.GetArrayFromImage(seg_de_filtered)+ \
        sitk.GetArrayFromImage(filter_components_by_overlap(sitk.Cast(seg_aggVsInd_both>0, sitk.sitkUInt8),sitk.Cast(seg>1,sitk.sitkUInt8),0.01)
        )
    combinedIm = np.where(combinedIm <= 2, combinedIm, combinedIm - 1)

    sum_image = sitk.GetImageFromArray(combinedIm)
    sum_image.CopyInformation(t2)

    fn = os.path.join(out_path, case_id + "_mpa_f_label.nii.gz")
    sitk.WriteImage(sum_image, fn)
    print("  Done writing metastasis proliferation aggressive combined label:", fn)

    # create contact line from region where all three criteria of aggressiveness
    # are predicted.
    # Dilate label 3 by 3 mm and measure volume outside prostate segmentation
    radius = 3.0; #mm
    label3_mask = sitk.Cast(sum_image == 3, sitk.sitkUInt8)
    spacing = sum_image.GetSpacing()
    radius_vox = [max(1, round(radius / s)) for s in spacing]
    label3_dilated = sitk.BinaryDilate(label3_mask, radius_vox)
    label3_dilated_outside_seg = label3_dilated * sitk.Cast(seg < 1, sitk.sitkUInt8)

    fn = os.path.join(out_path, case_id + "_cln_label.nii.gz")
    sitk.WriteImage(label3_dilated_outside_seg , fn)
    print("  Done writing contact line:", fn)

    if filter_by_csPCA:
        if DEBUG:
            combined_proba_arr = np.stack([gene_proba_agg, gene_proba_ki, gene_proba_de], axis=-1).astype('float32')
            vector_proba_image = sitk.GetImageFromArray(combined_proba_arr, isVector=True)
            vector_proba_image.CopyInformation(t2)

            fn = os.path.join(out_path, case_id + "_mpa_f.nii.gz")
            sitk.WriteImage(vector_proba_image, fn) 

    if filter_by_csPCA:
        stats = {
            "pro": compute_label_volume(seg, [1,2,3]),
            "csp": compute_label_volume(seg, 3),
            "agg": compute_label_volume(seg_aggVsInd_filtered, 1),
            "prf": compute_label_volume(seg_ki67_filtered, 1),
            "met": compute_label_volume(seg_de_filtered, 1),
            "hi3": compute_label_volume(sum_image, 3),
            "hi1": compute_label_volume(sum_image, [2, 3]),
            "ind": compute_label_volume(sum_image, 1),
            "cln": compute_label_volume(label3_dilated_outside_seg, 1),
        }
        return stats


    return None

if __name__=="__main__":
    print("Unit test code to run pAres inference for one study.")
    parser = argparse.ArgumentParser(description='Run inference for all models')
    parser.add_argument('--t2', type=str, 
                        required=False,
                        default='./example_data/1003_0001_t2w.mha',
                        help='path to t2 image or a folder including multiple T2')
    parser.add_argument('--adc', type=str, 
                        required=False,
                        default='./example_data/1003_0001_adc.mha',
                        help='path to the adc image')
    parser.add_argument('--output', '-o', type=str,
                        default='./results/1003',
                        required=False, 
                        help='folder where to put all the results')
    parser.add_argument('--case_id', '-d', type=str,
                        default='1003',
                        required=False, 
                        help='suffix to add to files')
    parser.add_argument('--proba_threshold', '-p', type=float,
                        default=0.1,
                        required=False, 
                        help='what probability to use to threshold the output')
    parser.add_argument('--psa', '-ps', type=float,
                        default=8.3,
                        required=False, 
                        help='the psa of the patient needed to compute the risk for extraprostatic extension and biochemical reccurence.')
        
    args = parser.parse_args()

    model_paths = {'csPCA':"models/Dataset202_BxMR_withRegions_T2_ADC/nnUNetTrainer__nnUNetPlans__3d_fullres/",
                   'aggInd':"models/Dataset203_CaAggInd_i4ch_oIndAggCh_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/",
                   'KI67':"models/Dataset361_12342_MKI67_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres",
                   'Metastasis':"models/Dataset309_Decipher_som_fold0/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
                  }

    if not os.path.exists('models'):
        print("ERROR: Can't find the AI models. Please create the folder 'models', to include the four trained models:",
        "\n  1) Dataset202_BxMR_withRegions_T2_ADC",
        "\n  2) Dataset203_CaAggInd_i4ch_oIndAggCh_fold0",
        "\n  3) Dataset361_12342_MKI67_fold0",
        "\n  4) Dataset309_Decipher_som_fold0.\nThey are available online and from mrusu@stanford.edu!\nExiting now!")
        exit()
    
    for p in model_paths.keys():
        if not os.path.exists(model_paths[p]):
            print("Can't find a model folder: ", model_paths[p])
            exit ()


    print("Testing patient level models")
    # compute the epe probability
    th_epe = 0.4873275104221620
    feats = [[43736.438, 8.3, 1362.744]]
    epe_model_path = os.path.join('models/patient_level/rf2_epe.pkl')        
    epe_scalar_path = os.path.join('models/patient_level/scaler_epe.pkl')        
    epe_proba = get_patient_proba(epe_scalar_path, epe_model_path, feats)    

    feats = [[8.3, 3859.812, 6382.152]]
    # compute bcr probability
    th_bcr = 0.10545787330199500
    brc_model_path = os.path.join('models/patient_level/rf2_bcr.pkl')        
    bcr_scalar_path = os.path.join('models/patient_level/scaler_bcr.pkl')        
    bcr_proba = get_patient_proba(bcr_scalar_path, brc_model_path, feats)

    print("epe probability:", epe_proba[0], "-> Binary EPE Status: ", float(epe_proba[0])>th_epe, 
          "\nbcr probability:", bcr_proba[0] ,"-> Binary BRC Status: ", float(bcr_proba[0])>th_bcr)    

    stats = run_one_case(args.t2, args.adc, args.output, model_paths, args.case_id, args.proba_threshold)



    if stats is not None:
        print("Features computed from the different regions")
        for s in stats.keys():
            print(f"{s} : {stats[s]:9.3f}")

    print("Testing patient level models")
    print("epe probability:", epe_proba[0], "-> Binary EPE Status: ", float(epe_proba[0])>th_epe, 
           "\nbcr_probability:", bcr_proba[0] ,"-> Binary BRC Status: ", float(bcr_proba[0])>th_bcr)    
    # compute the epe probability
    th_epe = 0.4873275104221620
    feats = [[stats['pro'], args.psa, stats['cln']]]
    epe_model_path = os.path.join('models/patient_level/rf2_epe.pkl')        
    epe_scalar_path = os.path.join('models/patient_level/scaler_epe.pkl')        
    epe_proba = get_patient_proba(epe_scalar_path, epe_model_path, feats)    

    feats = [[args.psa, stats['hi3'], stats['prf']]]
    # compute bcr probability
    th_bcr = 0.10545787330199500
    brc_model_path = os.path.join('models/patient_level/rf2_bcr.pkl')        
    bcr_scalar_path = os.path.join('models/patient_level/scaler_bcr.pkl')        
    bcr_proba = get_patient_proba(bcr_scalar_path, brc_model_path, feats)

    print("Computed EPE probability:", epe_proba[0], "-> Binary EPE Status: ", float(epe_proba[0])>th_epe, 
          "\nComputed BRC probability:", bcr_proba[0] ,"-> Binary BRC Status: ", float(bcr_proba[0])>th_bcr)    
