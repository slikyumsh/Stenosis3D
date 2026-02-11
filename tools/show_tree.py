import nibabel as nib
import napari
import numpy as np

vol = nib.load("exports/100_mask_only.nii.gz").get_fdata()
viewer = napari.Viewer(ndisplay=3)
viewer.add_labels((vol > 0).astype(np.uint8))
napari.run()
