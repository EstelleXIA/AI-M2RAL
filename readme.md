This demo repository is for AI-based multi-faceted integration of multi-phase radiomics and laboratory data for the renal mass classification (AI-M2RAL) model.

## Environment

You should use `./environment.yaml` to create a conda environment:

```
conda env create -n classification -f environment.yaml
```

## Data Preprocessing

#### 1. Data collection

The CT data should included multi-phase images (non-contrast, arterial, venous, and delayed phases), and then being converted to NIFTI format.

#### 2. Automatic tumor segmentation

Then, for the automatic segmentation of kidney tumors, we refer to the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). We used pretrained weights on KiTS to automatically segment renal tumors from the 3D CT scans on arterial phase, which can be downloaded [here](https://zenodo.org/record/3734294#.YvXiWnZBw2z).

#### 3. Cross-phase registration

Deformable registration based on advanced normalization tools ([ANTs](https://github.com/ANTsX/ANTs)) was performed to align scans of the non-contrast, venous, and delayed phases to the arterial phase. You can refer to the processing demo in `./data_preprocessing/1_ants_registrate.py`.

#### 4. Crop/pad the ROIs

The 3D region of interest (ROI) is extracted from the registered four-phase scans according to the tumor mask extracted from the arterial phase in step 2. You can refer to the processing demo in `./data_preprocessing/2_crop_roi.py` (crop the ROI region), `./data_preprocessing/3_pad_roi.py `(pad the region to a fix region of 128×128×192), `./data_preprocessing/4_get_has_tumor_3d.py ` (select the tumor side),  `./data_preprocessing/5_get_tumor.py ` (generate mask that only include tumor region), and `./data_preprocessing/6_get_largest_2d.py ` (generate 2d slice with largest tumor).

#### 5. Pyradiomics extraction

After getting the cropped CT ROIs and tumor masks, [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) toolkit is used for radiomics feature extraction. You can refer to the processing demo in `./data_preprocessing/7_radiomics_extraction.py`.

#### 6. ResNet feature extraction

The 2D slice with the largest tumor in the axial view was selected to extract morphological features using the ResNet-34 model. You can refer to the processing demo in `./data_preprocessing/8_extract_resnet_2d.py`.

## Model Construction

We proposed the AI-based multi-faceted integration of multi-phase radiomics and laboratory data for renal tumor classification (AI-M2RAL). The multi-modal deep learning method, is designed to subtype renal tumors into three categories: benign (includes renal oncocytoma and angiomyolipoma), non-clear cell carcinoma (includes papillary and chromophobe renal carcinoma), and clear cell carcinoma.

For training the model, run:

```
python ./model_construction/train.py --model_type M2RAL
```

You can try the model without lab data:

```
python ./model_construction/train.py --model_type M2RAD
```

Also the model proposed by [Dai et al.](https://doi.org/10.1148/radiol.232178):

```
python ./model_construction/train.py --model_type Dai
```

