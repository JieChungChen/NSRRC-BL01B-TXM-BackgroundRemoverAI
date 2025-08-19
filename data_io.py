import os, glob, olefile, struct
import numpy as np
from PIL import Image
from math import sqrt
from utils import split_mosaic


def read_txm_raw(filename, mode):
    assert mode in ['tomo', 'mosaic', 'single'], 'wrong mode !'

    ole = olefile.OleFileIO(filename)
    n_img = len([entry for entry in ole.listdir() if entry[0] in ['ImageData1', 'ImageData2']])
    metadata = read_ole_metadata(ole, mode, n_img)

    if mode == 'tomo':
        raw_imgs = np.empty((metadata["number_of_images"],
                            metadata["image_height"],
                            metadata["image_width"]),
                            dtype=_get_ole_data_type(metadata))

        for i, idx in enumerate(range(metadata["number_of_images"])):
            img_string = "ImageData{}/Image{}".format(int(np.ceil((idx + 1) / 100.0)), int(idx + 1))
            raw_imgs[i] = _read_ole_image(ole, img_string, metadata)
        ole.close()
        raw_imgs = np.flip(raw_imgs, axis=1)
        angles = metadata['thetas'][:n_img]
        metadata['thetas'] = np.around(angles, decimals=1)

    if mode == 'mosaic':
        stream = ole.openstream("ImageData1/Image1")
        data = stream.read()
        data_type = _get_ole_data_type(metadata)
        data_type = data_type.newbyteorder('<')
        image = np.reshape(np.frombuffer(data, data_type), (metadata["image_height"],  metadata["image_width"]))
        ole.close()
        raw_imgs = split_mosaic(image, metadata['mosaic_row'], metadata['mosaic_column'])
        raw_imgs = np.flip(raw_imgs, axis=1)

    if mode == 'single':
        stream = ole.openstream("ImageData1/Image1")
        data = stream.read()
        data_type = _get_ole_data_type(metadata)
        data_type = data_type.newbyteorder('<')
        image = np.reshape(np.frombuffer(data, data_type), (metadata["image_height"],  metadata["image_width"]))
        image = np.flip(image, axis=0)
        ole.close()

    reference = metadata['reference']
    metadata.pop('reference', None)
    metadata.pop('reference_data_type', None)
    metadata.pop('data_type', None)

    if mode == 'tomo' or mode == 'mosaic':
        return raw_imgs, metadata, reference
    if mode == 'single':
        return image, metadata


def read_ole_metadata(ole, mode, n_img=None):
    """
    Read metadata from an xradia OLE file (.xrm, .txrm, .txm).

    Parameters
    ----------
    ole : OleFileIO instance
        An ole file to read from.

    Returns
    -------
    tuple
        A tuple of image metadata.
    """

    if n_img is not None:
        number_of_images = n_img
    else:
        number_of_images = _read_ole_value(ole, "ImageInfo/NoOfImages", "<I")
    metadata = {
        'number_of_images': number_of_images,
        'image_width': _read_ole_value(ole, 'ImageInfo/ImageWidth', '<I'),
        'image_height': _read_ole_value(ole, 'ImageInfo/ImageHeight', '<I'),
        'pixel_size': _read_ole_value(ole, 'ImageInfo/pixelsize', '<f'),
        'data_type': _read_ole_value(ole, 'ImageInfo/DataType', '<1I'),
        'reference_filename': _read_ole_value(ole, 'ImageInfo/referencefile', '<260s'),
        'reference_data_type': _read_ole_value(ole, 'referencedata/DataType', '<1I'),
                }
    
    if mode == 'tomo':
        metadata['thetas'] = _read_ole_arr(ole, 'ImageInfo/Angles', "<{0}f".format(number_of_images))
    elif mode == 'mosaic':
        metadata['mosaic_column'] = _read_ole_value(ole, 'ImageInfo/MosiacColumns', '<I')
        metadata['mosaic_row'] = _read_ole_value(ole, 'ImageInfo/MosiacRows', '<I')
    
    ref_path = _read_ole_value(ole, 'ImageInfo/referencefile', '<260s')
    if ref_path is not None:
        ref_path = ref_path.strip(b'\x00').decode()
        metadata['reference_filename'] = ref_path.split('\\')[-1]
    
    if ole.exists('ReferenceData/Image'):
        reference = _read_ole_image(ole, 'ReferenceData/Image', metadata, metadata['reference_data_type'], is_ref=True)
    else:
        reference = None
    metadata['reference'] = reference
    return metadata


def _get_ole_data_type(metadata, datatype=None):
    # 10 float; 5 uint16 (unsigned 16-bit (2-byte) integers)
    if datatype is None:
        datatype = metadata["data_type"]
    if datatype == 10:
        return np.dtype(np.float32)
    elif datatype == 5:
        return np.dtype(np.uint16)
    else:
        raise Exception("Unsupport XRM datatype: %s" % str(datatype))


def _read_ole_struct(ole, label, struct_fmt):
    """
    Reads the struct associated with label in an ole file
    """
    value = None
    if ole.exists(label):
        stream = ole.openstream(label)
        data = stream.read()
        if label == 'ImageInfo/Angles':
            value = struct.unpack("<{0}f".format(len(data)//4), data)
        else:
            value = struct.unpack(struct_fmt, data)
    return value


def _read_ole_value(ole, label, struct_fmt):
    """
    Reads the value associated with label in an ole file
    """
    value = _read_ole_struct(ole, label, struct_fmt)
    if value is not None:
        value = value[0]
    return value


def _read_ole_arr(ole, label, struct_fmt):
    """
    Reads the numpy array associated with label in an ole file
    """
    arr = _read_ole_struct(ole, label, struct_fmt)
    if arr is not None:
        arr = np.array(arr)
    return arr


def _read_ole_image(ole, label, metadata, datatype=None, is_ref=False):
    stream = ole.openstream(label)
    data = stream.read()
    data_type = _get_ole_data_type(metadata, datatype)
    data_type = data_type.newbyteorder('<')
    image = np.frombuffer(data, data_type)
    if is_ref:
        s = int(sqrt(len(image)))
        img_size = (s, s)
    else:
        img_size = (metadata["image_height"], metadata["image_width"])
    image = np.reshape(image, img_size)
    return image