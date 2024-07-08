import matplotlib.pyplot as plt


def display_dicoms_grid(dicoms: list, rows: int = 2, cols: int = 3):
    fig = plt.figure(figsize=(10,10))
    
    for idx, ds in enumerate(dicoms):
        if idx >= rows*cols:
            break
        fig.add_subplot(rows, cols, idx+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

def display_dicom(ds):
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 