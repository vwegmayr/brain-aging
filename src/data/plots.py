import matplotlib.pyplot as plt
import nibabel as nib


def display_image_path(path):
    img = nib.load(path)
    dat = img.get_data()
    if len(dat.shape) == 4:
        dat = dat[:,:,:, 0]
    imshowargs = {
        'interpolation': 'nearest',
    }

    def forceAspect(ax, aspect=1):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

    ax = plt.subplot(131)
    ax.imshow(dat[dat.shape[0]/2,:,:], **imshowargs)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    forceAspect(ax)
    
    ax = plt.subplot(132)
    ax.imshow(dat[:,dat.shape[1]/2,:], **imshowargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    forceAspect(ax)
    
    ax = plt.subplot(133)
    ax.imshow(dat[:,:, dat.shape[2]/2], **imshowargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    forceAspect(ax)
    
    plt.tight_layout()
    plt.show()
    return dat