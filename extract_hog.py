# Python Script for extract HOG feature from Orange Datatable that contains image data
# This is an alternative of Orange Image Analytics add-on to extract feature from image data

def python_script(in_data, in_learner, in_classifier, in_object):
    from Orange.data import Table
    import numpy as np
    from skimage.feature import hog
    from skimage import io
    from Orange.data import Domain, ContinuousVariable, DiscreteVariable
    from skimage.transform import resize
    import os

    # the path of the image folder
    image_path = "C:\\Users\\ekoru\\Source-Code\\Orange-HOG\\Diode\\"

    # Prepare the out_data table
    # the number of feature
    feature_len = 72
    # Create new list of ContinuousVariable with the total number of feature_len
    list_variable = [ContinuousVariable("col" + str(i)) for i in range(feature_len)]
    # add DiscreteVariable with name 'target' and values 'OK' and 'NG' to list_variable
    list_variable.append(DiscreteVariable("target", values=["OK", "NG"]))
    # create new domain with list_variable
    domain = Domain(list_variable)
    data_image = in_data.metas
    arr = np.zeros((len(in_data), feature_len + 1))

    for i in range(0, len(in_data)):
        print("Data image: ", image_path + data_image[i][1])
        # get filename from data table
        filename = image_path + data_image[i][1]
        image = io.imread(filename)
        image_resized = resize(image, (50, 50), anti_aliasing=True)
        fd, hog_image = hog(
            image_resized,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )
        arr[i][0:feature_len] = fd
        arr[i][feature_len] = in_data.Y[i]
    out_data = Table.from_numpy(domain, X=arr)

    out_learner = in_learner
    out_classifier = in_classifier
    out_object = in_object

    return out_data, out_learner, out_classifier, out_object
