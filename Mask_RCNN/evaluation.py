import cv2 as cv
from skimage import io

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '.'
WORKING_DIRECTORY = '../generated/'
real_test_dir = WORKING_DIRECTORY + 'real_test/'
HEIGHT = 1664
WIDTH = 3328

# Load the images to superimpose
image_paths = ['171.png', '212.png', 'aku22-93.png']
for image_path in image_paths:
    img1 = cv.imread(real_test_dir + 'segment_results/masked-' + image_path)
    img2 = cv.imread(real_test_dir + 'lesion_results/lesion-' + image_path)
    # img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    # img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    # Resize img2 to match the size of img1
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    # Blend the two images together using cv2.addWeighted
    superimposed_img = cv.addWeighted(img1, 0.5, img2, 1.0, 0)
    # Display the resulting image using skimage.io.imshow
    io.imshow(superimposed_img)
    io.show()
    cv.imwrite(real_test_dir + 'superimposed-' + image_path, superimposed_img)

# Save as a 2x2 image as a table
for image_path in image_paths:
    img1 = cv.imread(real_test_dir + image_path)
    img2 = cv.imread(real_test_dir + 'lesion_results/lesion-' + image_path)
    img3 = cv.imread(real_test_dir + 'segment_results/masked-' + image_path)
    img4 = cv.imread(real_test_dir + 'superimposed-' + image_path)
    # Resize the images to the same size
    img1 = cv.resize(img1, (WIDTH, HEIGHT))
    img2 = cv.resize(img2, (WIDTH, HEIGHT))
    img3 = cv.resize(img3, (WIDTH, HEIGHT))
    img4 = cv.resize(img4, (WIDTH, HEIGHT))
    # Create a 2x2 table with all 4 images
    top_row = cv.hconcat([img1, img2])
    bottom_row = cv.hconcat([img3, img4])
    final_image = cv.vconcat([top_row, bottom_row])
    io.imshow(final_image)
    io.show()
    cv.imwrite(real_test_dir + 'final-' + image_path, final_image)

