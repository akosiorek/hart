"""Resize images functions

If called directly resizes a whole tree with keeping the the tree structutre of the source in destination
"""
import cv2
import os
import argparse


def resize_image(im, height=None, width=None, inter=cv2.INTER_LINEAR):
    """
    resize image and keeps aspect ratio if only one of height or width is supplied
    :param im: image
    :type im: np.array
    :param height: the desired height of the resized image if None [default] the height is calculated to keep aspect 
    ratio with width
    :type height: int
    :param width: the desired width of the resized image if None [default] the width is calculated to keep aspect ratio 
    with height
    :type width: int
    :param inter: interpolation method 
    :return: resized image
    """
    orig_height, orig_width = im.shape[:2]

    if width is None and height is None:
        raise TypeError('{} requires on of the arguments: height or width'.format(__name__))
    elif width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(orig_height)
        dim = (int(worig_width * r), height)
    elif height is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = width / float(orig_width)
        dim = (width, int(orig_width * r))
    else:
        dim = (width, height)
    return cv2.resize(im, dim, interpolation=inter)


def resize_im_file_and_save(src, dst, height=None, width=None, inter=cv2.INTER_LINEAR):
    """
    resize image file and write it to destination. keeps aspect ratio if only one of height or width is supplied
    :param src: the source image path
    :type src: str
    :param dst: the destination image path
    :type dst: str
    :param height: the desired height of the resized image if None [default] the height is calculated to keep aspect 
    ratio with width
    :type height: int
    :param width: the desired width of the resized image if None [default] the width is calculated to keep aspect ratio 
    with height
    :type width: int
    :param inter: interpolation method 
    :return: resized image
    """
    im = cv2.imread(src)
    resized = resize_image(im, height, width, inter)
    cv2.imwrite(dst, resized)


def resize_images_tree(src, dst, h=None, w=None, inter=cv2.INTER_LINEAR):
    """
    resizes images form source wile keeping the the tree structutre of the source in destination
    :param src: the source path
    :type src: str
    :param dst: the destination path
    :type dst: str
    :param height: the desired height of the resized image if None [default] the height is calculated to keep aspect 
    ratio with width
    :type height: int
    :param width: the desired width of the resized image if None [default] the width is calculated to keep aspect ratio 
    with height
    :type width: int
    :param inter: interpolation method 
    :return: resized image 
    """
    for root_, dirs, files in os.walk(src):
        for dir in dirs:
            rel_dir = os.path.relpath(os.path.join(root_, dir), src)
            dst_dir = os.path.join(dst, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if not any([file.endswith(f_type) for f_type in ['jpg', 'png', 'bmp', 'gif', 'jpeg']]):
                continue

            orig_file = os.path.join(root_, file)
            if not os.path.getsize(orig_file):
                continue

            rel_dir = os.path.relpath(root_, src)
            new_dst = os.path.join(dst, rel_dir)
            new_file = os.path.join(new_dst, file)

            resize_im_file_and_save(orig_file, new_file, h, w, inter)

            print('resized {} saved at {}'.format(orig_file, new_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image resizer')

    parser.add_argument('-s', action='store', dest='input_dir', help='source/input directory')
    parser.add_argument('-d', action='store', dest='output_dir', help='destination/output directory')
    parser.add_argument('-m', action='store', type=int, dest='h', help='output height')
    parser.add_argument('-n', action='store', type=int, dest='w', help='output width')

    arguments = parser.parse_args()

    if arguments.input_dir is None:
        raise AttributeError('-s input_dir is missing')

    if arguments.output_dir is None:
        raise AttributeError('-d output_dir is missing')

    if arguments.w is None and arguments.h is None:
        raise AttributeError('missong either -m or -n')

    input_dir = arguments.input_dir
    output_dir = arguments.output_dir

    os.makedirs(output_dir, exist_ok=True)

    h = arguments.h
    w= arguments.w

    resize_images_tree(input_dir, output_dir, h, w)
    reize image file and write to dst
