# -*- encoding: utf-8 -*-
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

def resize_image_itk(itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
    newSpacing = np.array(newSpacing, float)
    # originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)  # 将输出的大小、原点、间距和方向设置为itkimage
    resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
    resampler.SetSize(newSize.tolist())  # 设置输出图像大小
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled 


# This function is adapted from OpenKBP official codes, https://github.com/ababier/open-kbp
def load_csv_file(file_name):
    # Load the file as a csv
    loaded_file_df = pd.read_csv(file_name, index_col=0)

    # If the csv is voxel dimensions read it with numpy
    if 'voxel_dimensions.csv' in file_name:
        loaded_file = np.loadtxt(file_name)
    # Check if the data has any values
    elif loaded_file_df.isnull().values.any():
        # Then the data is a vector, which we assume is for a mask of ones
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:
        # Then the data is a matrix of indices and data points
        loaded_file = {'indices': np.array(loaded_file_df.index).squeeze(),
                       'data': np.array(loaded_file_df['data']).squeeze()}

    return loaded_file


# Transform numpy array(Z * H * W) to NITFI(nii) image
# Default image direction (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Default origin (0.0, 0.0, 0.0)
def np2NITFI(image, spacing):
    image_nii = sitk.GetImageFromArray(image)
    image_nii.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    image_nii.SetSpacing(tuple(spacing))
    image_nii.SetOrigin((0.0, 0.0, 0.0))

    return image_nii


if __name__ == '__main__':
    #source_dir = 'E:/RTDosePrediction/RTDosePrediction/Data/open-kbp-master/provided-data'
    #save_dir = 'E:/RTDosePrediction/RTDosePrediction/Data/OpenKBP_spacing'
    
    source_dir = 'YOUR_ROOT/Data/open-kbp-master/provided-data'
    save_dir = 'YOUR_ROOT/Data/OpenKBP_spacing'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    list_patient_dirs = []
    for sub_dir in ['train-pats', 'validation-pats', 'test-pats']:
        for patient_id in os.listdir(source_dir + '/' + sub_dir):
            list_patient_dirs.append(source_dir + '/' + sub_dir + '/' + patient_id)

    print(list_patient_dirs)

    for patient_dir in list_patient_dirs:
        # Make dir for each patient
        patient_id = patient_dir.split('/')[-1]
        save_patient_path = save_dir + '/' + patient_id
        if not os.path.exists(save_patient_path):
            os.mkdir(save_patient_path)

        # Spacing
        spacing = load_csv_file(patient_dir + '/voxel_dimensions.csv')

        # CT
        CT_csv = load_csv_file(patient_dir + '/ct.csv')
        CT = np.zeros((128, 128, 128), dtype=np.int16)
        indices_ = np.int64(CT_csv['indices'])
        data_ = np.int16(CT_csv['data'])
        np.put(CT, indices_, data_)
        CT = CT - 1024

        # Data in OpenKBP dataset is (h, w, -z) or (y, x, -z)
        CT = CT[:, :, ::-1].transpose([2, 0, 1])
        CT_nii = np2NITFI(CT, spacing)
        CT_nii = resize_image_itk(CT_nii, [3.906,3.906, 2.5], spacing)
        CT = sitk.GetArrayFromImage(CT_nii)

        if CT.shape[0] >= 128 and CT.shape[1] >= 128:
            CT = CT[CT.shape[0]//2-64:CT.shape[0]//2+64, CT.shape[1]//2-64:CT.shape[1]//2+64, CT.shape[2]//2-64:CT.shape[2]//2+64]
        elif CT.shape[0] < 128 and CT.shape[1] < 128:
            tmp = np.zeros((128,128,128))
            tmp[64-CT.shape[0]//2: 64-CT.shape[0]//2+CT.shape[0], 64-CT.shape[1]//2: 64-CT.shape[1]//2+CT.shape[1], 64-CT.shape[2]//2: 64-CT.shape[2]//2+CT.shape[2]] = CT
            CT = tmp
        elif CT.shape[0] >= 128 and CT.shape[1] < 128:
            tmp = np.zeros((128,128,128))
            tmp[:, 64-CT.shape[1]//2: 64-CT.shape[1]//2+CT.shape[1], 64-CT.shape[2]//2: 64-CT.shape[2]//2+CT.shape[2]] = CT[CT.shape[0]//2-64:CT.shape[0]//2+64, :, :]
            CT = tmp
        elif CT.shape[0] < 128 and CT.shape[1] >= 128:
            tmp = np.zeros((128,128,128))
            tmp[64-CT.shape[0]//2: 64-CT.shape[0]//2+CT.shape[0], :, :] = CT[:, CT.shape[1]//2-64:CT.shape[1]//2+64, CT.shape[2]//2-64:CT.shape[2]//2+64]
            CT = tmp
        print(CT.shape)
        CT_nii = np2NITFI(CT, [3.906,3.906, 2.5])

        sitk.WriteImage(CT_nii, save_patient_path + '/CT.nii.gz')

        # Dose
        dose_csv = load_csv_file(patient_dir + '/dose.csv')
        dose = np.zeros((128, 128, 128), dtype=np.float32)
        indices_ = np.int64(dose_csv['indices'])
        data_ = np.float32(dose_csv['data'])
        np.put(dose, indices_, data_)

        dose = dose[:, :, ::-1].transpose([2, 0, 1])
        dose_nii = np2NITFI(dose, spacing)
        dose_nii = resize_image_itk(dose_nii, [3.906,3.906, 2.5], spacing)
        dose = sitk.GetArrayFromImage(dose_nii)
        
        if dose.shape[0] >= 128 and dose.shape[1] >= 128:
            dose = dose[dose.shape[0]//2-64:dose.shape[0]//2+64, dose.shape[1]//2-64:dose.shape[1]//2+64, dose.shape[2]//2-64:dose.shape[2]//2+64]
        elif dose.shape[0] < 128 and dose.shape[1] < 128:
            tmp = np.zeros((128,128,128))
            tmp[64-dose.shape[0]//2: 64-dose.shape[0]//2+dose.shape[0], 64-dose.shape[1]//2: 64-dose.shape[1]//2+dose.shape[1], 64-dose.shape[2]//2: 64-dose.shape[2]//2+dose.shape[2]] = dose
            dose = tmp
        elif dose.shape[0] >= 128 and dose.shape[1] < 128:
            tmp = np.zeros((128,128,128))
            tmp[:, 64-dose.shape[1]//2: 64-dose.shape[1]//2+dose.shape[1], 64-dose.shape[2]//2: 64-dose.shape[2]//2+dose.shape[2]] = dose[dose.shape[0]//2-64:dose.shape[0]//2+64, :, :]
            dose = tmp
        elif dose.shape[0] < 128 and dose.shape[1] >= 128:
            tmp = np.zeros((128,128,128))
            tmp[64-dose.shape[0]//2: 64-dose.shape[0]//2+dose.shape[0], :, :] = dose[:, dose.shape[1]//2-64:dose.shape[1]//2+64, dose.shape[2]//2-64:dose.shape[2]//2+64]
            dose = tmp

        print(dose.shape)
        dose_nii = np2NITFI(dose, [3.906,3.906, 2.5])
        sitk.WriteImage(dose_nii, save_patient_path + '/dose.nii.gz')

        # OARs
        for structure_name in ['PTV70',
                               'PTV63',
                               'PTV56',
                               'possible_dose_mask',
                               'Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible']:
            structure_csv_file = patient_dir + '/' + structure_name + '.csv'
            if os.path.exists(structure_csv_file):
                structure_csv = load_csv_file(structure_csv_file)
                structure = np.zeros((128, 128, 128), dtype=np.uint8)
                np.put(structure, structure_csv, np.uint8(1))

                structure = structure[:, :, ::-1].transpose([2, 0, 1])
                structure_nii = np2NITFI(structure, spacing)
                structure_nii = resize_image_itk(structure_nii, [3.906,3.906, 2.5], spacing)
                structure = sitk.GetArrayFromImage(structure_nii)
                if structure.shape[0] >= 128 and structure.shape[1] >= 128:
                    structure = structure[structure.shape[0]//2-64:structure.shape[0]//2+64, structure.shape[1]//2-64:structure.shape[1]//2+64, structure.shape[2]//2-64:structure.shape[2]//2+64]
                elif structure.shape[0] < 128 and structure.shape[1] < 128:
                    tmp = np.zeros((128,128,128))
                    tmp[64-structure.shape[0]//2: 64-structure.shape[0]//2+structure.shape[0], 64-structure.shape[1]//2: 64-structure.shape[1]//2+structure.shape[1], 64-structure.shape[2]//2: 64-structure.shape[2]//2+structure.shape[2]] = structure
                    structure = tmp
                elif structure.shape[0] >= 128 and structure.shape[1] < 128:
                    tmp = np.zeros((128,128,128))
                    tmp[:, 64-structure.shape[1]//2: 64-structure.shape[1]//2+structure.shape[1], 64-structure.shape[2]//2: 64-structure.shape[2]//2+structure.shape[2]] = structure[structure.shape[0]//2-64:structure.shape[0]//2+64, :, :]
                    structure = tmp
                elif structure.shape[0] < 128 and structure.shape[1] >= 128:
                    tmp = np.zeros((128,128,128))
                    tmp[64-structure.shape[0]//2: 64-structure.shape[0]//2+structure.shape[0], :, :] = structure[:, structure.shape[1]//2-64:structure.shape[1]//2+64, structure.shape[2]//2-64:structure.shape[2]//2+64]
                    structure = tmp
                print(structure.shape)
                structure_nii = np2NITFI(structure, [3.906,3.906, 2.5])
                sitk.WriteImage(structure_nii, save_patient_path + '/' + structure_name + '.nii.gz')

        print(patient_id + ' done !')



