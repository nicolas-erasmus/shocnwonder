import numpy as np
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse


def read_fits_file(file_path):
    hdulist = fits.open(file_path)
    data_cube = hdulist[0].data
    header = hdulist[0].header
    
    hdulist.close()
    return data_cube, header

def extract_stars(image, index):
    # Background subtraction
    bkg = sep.Background(image)

    # Determine the size of the border as a percentage of the image dimensions
    border_percentage = 2  # Adjust this percentage as needed
    X, Y = np.array(image.shape) * border_percentage // 100

    # Create a mask excluding a border along the edges
    mask = np.ones_like(image, dtype=bool)
    mask[:X, :] = False
    mask[-X:, :] = False
    mask[:, :Y] = False
    mask[:, -Y:] = False

    # Invert the mask (True where you want to extract sources)
    mask = ~mask

    # Extract sources with the mask and minimum pixel size requirement
    objects = sep.extract(image - bkg, 1.5, err=bkg.globalrms, mask=mask, minarea=20)

    # Display the image with ellipses indicating the sources
    fig, ax = plt.subplots()
    m, s = np.mean(image), np.std(image)
    im = ax.imshow(image, cmap='gray', vmin=m-s, vmax=m+s, origin='lower')

    # Fit Gaussian and extract FWHM for each source
    fwhm_x_values = []
    fwhm_y_values = []
    for i in range(len(objects)):
        half_size = 50
        subimage = image[int(objects['y'][i]) - half_size:int(objects['y'][i]) + half_size,
                         int(objects['x'][i]) - half_size:int(objects['x'][i]) + half_size]
        try:
            fwhm_x, fwhm_y = fit_2d_gaussian(subimage)
            print(fwhm_x, fwhm_y)
        except:
            print("could not fit, set fwhm to nan")
            fwhm_x, fwhm_y = np.nan, np.nan

        fwhm_x_values.append(fwhm_x)
        fwhm_y_values.append(fwhm_y)

        # Plot ellipses instead of circles
        ellipse = Ellipse((objects['x'][i], objects['y'][i]), fwhm_x, fwhm_y, edgecolor='red', facecolor='none')
        ax.add_patch(ellipse)

    ax.set_title('Image with Detected Sources')
    plt.colorbar(im)

    # Save the figure as a PNG file
    plt.savefig(f'sources_{index}.png')

    # Close the figure to avoid displaying it
    plt.close()

    return objects, fwhm_x_values, fwhm_y_values

def gaussian2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(- (a * (x - xo)**2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo)**2))
    return g.ravel()

def fit_2d_gaussian(image):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xy = (x, y)

    initial_guess = (np.max(image), np.argmax(image) % image.shape[1], np.argmax(image) // image.shape[1],
                     1.0, 1.0, 0.0, np.min(image))

    popt, _ = curve_fit(gaussian2d, xy, image.ravel(), p0=initial_guess)

    fwhm_x = 2.355 * popt[3]  # FWHM in x direction
    fwhm_y = 2.355 * popt[4]  # FWHM in y direction

    return fwhm_x, fwhm_y


def plot_flux_radius(objects, fwhm_x_values, fwhm_y_values, index):
    # Calculate the distance of each source from the center of the image
    center_x, center_y = np.median(objects['x']), np.median(objects['y'])
    distances = np.sqrt((objects['x'] - center_x)**2 + (objects['y'] - center_y)**2)

    # Plot FWHM as a function of distance from the center
    fig, ax = plt.subplots()
    ax.scatter(np.multiply(distances,(plate_scale*x_bin)), np.multiply(fwhm_x_values,(plate_scale*x_bin)), c='red', marker='o', label='fwhm_x')
    ax.scatter(np.multiply(distances,(plate_scale*y_bin)), np.multiply(fwhm_y_values,(plate_scale*y_bin)), c='blue', marker='o', label='fwhm_y')
    
    ax.set_xlabel('Distance from Center (arcseconds)')
    ax.set_ylabel('FWHM (arcseconds)')
    ax.set_ylim(0,4)
    ax.legend()

    # Save the figure as a PNG file
    plt.savefig(f'fwhm_{index}.png')
    
def main(fits_file_path):
    global x_bin 
    global y_bin
    global plate_scale
    plate_scale = 0.16  # arcsec per unbinned pixel
    
    data_cube, header = read_fits_file(fits_file_path)
    
    x_bin = float(header["HBIN"])
    y_bin = float(header["VBIN"])

    for i in range(data_cube.shape[0]):
        current_image = data_cube[i].astype(float)
        print(f"Processing image {i + 1}/{data_cube.shape[0]}")

        objects, fwhm_x_values, fwhm_y_values = extract_stars(current_image,i)

        # Plotting FWHM as a function of radial distance for each image
        plot_flux_radius(objects, fwhm_x_values, fwhm_y_values,i)

if __name__ == "__main__":
    fits_file_path = "SHW_20240131.0007.fits"
    main(fits_file_path)
