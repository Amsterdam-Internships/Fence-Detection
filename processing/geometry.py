def pixel_to_viewpoint(pixel, image_width=8000, dtype=int):
    """
    Convert width in pixels to viewpoint in degrees
    """
    return dtype(360 * pixel / image_width)

def viewpoint_to_pixels(viewpoint, image_width=8000, dtype=int):
    """
    Convert viewpoint in degrees to width in pixels
    """
    return dtype(viewpoint / 360 * image_width)

def reindex(index, max):
    """
    Reindex out-of-bounds indices
    """
    if not index:
        return index

    if index < 0:
        index += max
    elif index > max:
        index -= max

    return index