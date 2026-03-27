from preprocessing.crop_pupil_camwfs import circular_mask, get_square_thumbnail_from_pupil


def make_circular_template(image, centerpt, rad):
    im_shape = image.shape
    subap_mask = circular_mask(im_shape, centerpt, rad)
    masked = image * subap_mask
    isolated_subap = get_square_thumbnail_from_pupil(masked)
    return isolated_subap