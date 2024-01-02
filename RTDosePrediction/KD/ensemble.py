import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))
from DataLoader.dataloader_OpenKBP_C3D import val_transform, read_data, pre_processing
from evaluate import *
import torch
from pre_models import teachers
from network_trainer import NetworkTrainer
from model import SearchedNet
import numpy as np
import SimpleITK as sitk


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


def flip_3d(input_, list_axes):
    if 'Z' in list_axes:
        input_ = input_[:, ::-1, :, :]
    if 'W' in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(teacher, input_, TTA_mode):
    list_prediction_B = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).cuda()
        
        prediction_B = teacher(augmented_input)[0]

        # Aug back to original order

        prediction_B = flip_3d(np.array(prediction_B.cpu().data[0, :, :, :, :]), list_flip_axes)

        list_prediction_B.append(prediction_B)

    return np.mean(list_prediction_B, axis=0)


def ensemble_result(trainer):
    list_patient_dirs = ['YOUR_ROOT/Data/RTDosePrediction/OpenKBP_C3D/pt_' + str(i) for i in range(201, 241)]
    save_path = 'YOUR_ROOT/Experiment/RTDosePrediction/Output/Ensemble_4'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        
        
        res = [[] for _ in range(100)]

        dose = np.zeros([100, 128, 128, 128])

        for i, teacher in enumerate(teachers):
                    
            teacher = teacher.cuda()
            teacher.eval()
            list_Dose_score = []

            for patient_dir in list_patient_dirs:
                patient_name = patient_dir.split('/')[-1]

                dict_images = read_data(patient_dir)
                list_images = pre_processing(dict_images)

                input_ = list_images[0]
                gt_dose = list_images[1]
                possible_dose_mask = list_images[2]

                TTA_mode = [[]]
                # Forward
                # [input_] = val_transform([input_])
                

                prediction_B = test_time_augmentation(teacher, input_, TTA_mode)

                # prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])

            
                # Post processing and evaluation
                prediction_B[np.logical_or(possible_dose_mask < 1, prediction_B < 0)] = 0
                dose[int(patient_name.split('_')[-1])-201] += prediction_B.squeeze()
                Dose_score = 70. * get_3D_Dose_dif(prediction_B.squeeze(0), gt_dose.squeeze(0),
                                                possible_dose_mask.squeeze(0))
             
                
                res[int(patient_name.split('_')[-1])-201].append(Dose_score)

                #print(res)

                list_Dose_score.append(Dose_score)

            print("Teacher {}: ".format(i), np.mean(list_Dose_score))
            # print(dose)
        
        dose /= len(teachers)

        list_Dose_score = []
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            gt_dose = list_images[1]
            possible_dose_mask = list_images[2]

            # Forward
            prediction_B = dose[int(patient_name.split('_')[-1])-201]

            # Post processing and evaluation
            # prediction_B[np.logical_or(possible_dose_mask < 1, prediction_B < 0)] = 0

            Dose_score = 70. * get_3D_Dose_dif(prediction_B, gt_dose.squeeze(0),
                                            possible_dose_mask.squeeze(0))
            
            
            res[int(patient_name.split('_')[-1])-201].append(Dose_score)

            list_Dose_score.append(Dose_score)

            prediction_B[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction_B < 0)] = 0
            prediction = 70. * prediction_B

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(patient_dir + '/possible_dose_mask.nii.gz')
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + patient_name):
                os.mkdir(save_path + '/' + patient_name)
            sitk.WriteImage(prediction_nii, save_path + '/' + patient_name + '/dose.nii.gz')

        
        print("Ensemble:", np.mean(list_Dose_score))




ensemble_result(NetworkTrainer)