########################################################################################
# 
# Hierarchical Attentive Recurrent Tracking
# Copyright (C) 2017  Adam R. Kosiorek, Oxford Robotics Institute, University of Oxford
# email:   adamk@robots.ox.ac.uk
# webpage: http://ori.ox.ac.uk
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 
########################################################################################

from itertools import izip

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation, rc
from matplotlib.patches import Rectangle

try:
    from IPython.display import HTML
except ImportError:
    pass

is_html_setup = False


def setup_html():
    rc('animation', html='html5')
    global is_html_setup
    is_html_setup = True


def wrap_animation(anim):
    if not is_html_setup:
        setup_html()

    return HTML(anim.to_html5_video())


def rect(bbox, c=None, facecolor='none', label=None, ax=None):
    r = Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2],
                  edgecolor=c, facecolor=facecolor, label=label)

    if ax is not None:
        ax.add_patch(r)
    return r


def plot_tracks(predicted, target, **kwargs):
    """Plots predicted and target 2D tracks on a 2D planes and
    tiles them together.

    :param predicted: np.ndarray of shape (T, N, 4)
    :param target: np.ndarray of shape (T, N, 4)
    :param kwargs: kwargs to plt.subplots
    :return: fig, axes
    """

    for k, v in zip(('sharex', 'sharey', 'figsize'), (True, True, (6, 6))):
        if k not in kwargs:
            kwargs[k] = v

    assert predicted.shape == target.shape, 'Shapes not equal: {} vs {}' \
        .format(predicted.shape, target.shape)

    fig, axes = plt.subplots(4, 3, **kwargs)
    for i, ax in enumerate(axes.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if i < predicted.shape[1]:
            ax.plot(predicted[:-1, i, 0], predicted[:-1, i, 1])
            ax.plot(target[1:, i, 0], target[1:, i, 1])
            ax.scatter(target[1, i, 0], target[1, i, 1], c='g')
            ax.scatter(target[-1, i, 0], target[-1, i, 1], c='r')
            ax.axis('equal')

    ax = axes[-1, -1]
    ax.plot([], label='predicted')
    ax.plot([], label='truth')
    ax.scatter([], [], c='g', label='start')
    ax.scatter([], [], c='r', label='end')
    ax.legend(loc=0, ncol=2)
    fig.tight_layout()
    return fig, axes


def anim_tracks_in_images(imgs, bboxes, glimpses=None, fps=30, save_as=None, dpi=300,
                          savefig_kwargs={'bbox_inches': 'tight'}, glimpse_loc='below', glimpses_per_row=2,
                          layout=dict(left=.025, right=.975, bottom=.025, top=.975, wspace=.025, hspace=.01)):
    """Animates images with bounding boxes.

    :param imgs: Images with shape (n_timesteps, height, width, channels).
    :param bboxes: Dictionary of legend_entry: bounding_box, where each
                        bounding box has the following shape:
                        (n_timesteps, y, x, height, width).
    :param fps: Frames per second for the created animation.
    :param save_as: Path under which the animation is saved. If given, no
        Animation is returned and the figure is immediately closed.
    :param dpi: Resolution of the saved animation.
    :param savefig_kwargs: kwargs passed to Animation.save.
    :return: pyplot.Animation if save_fig is None else None
    """

    if not isinstance(bboxes, dict):
        bboxes = {'bbox': bboxes}

    for v in bboxes.values():
        assert imgs.shape[0] == v.shape[0]

    num_frames = imgs.shape[0]

    if glimpses is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(glimpses, dict):
            glimpses = {'glimpse': glimpses}

        n_glimpse_rows = len(glimpses) / glimpses_per_row + (len(glimpses) % glimpses_per_row != 0)
        n_rows = 1 + n_glimpse_rows
        gs = gridspec.GridSpec(n_rows, glimpses_per_row, height_ratios=[2] + [1] * n_glimpse_rows)

        fig = plt.figure(figsize=(glimpses_per_row, n_rows))

        ax = plt.subplot(gs[0, :])
        glimpse_axes = np.empty((n_rows - 1, glimpses_per_row), dtype=object)
        for i in xrange(glimpse_axes.shape[0]):
            for j in xrange(glimpse_axes.shape[1]):
                glimpse_axes[i, j] = plt.subplot(gs[1 + i, j])

        glimpse_axes = glimpse_axes.flatten()
        glimpse_handles = tuple(
            (ga.imshow(g[0], cmap='gray', vmin=0., vmax=1.) for g, ga in zip(glimpses.values(), glimpse_axes)))

        for ga in glimpse_axes:
            ga.xaxis.set_visible(False)
            ga.yaxis.set_visible(False)

    handle = ax.imshow(imgs[0], cmap='gray')
    colors = 'rgbm'
    boxes = {}
    for c, (k, v) in zip(colors, bboxes.iteritems()):
        boxes[k] = Rectangle((v[0, 1], v[0, 0]), v[0, 3], v[0, 2],
                             edgecolor=c, facecolor='none', label=k)

        ax.add_patch(boxes[k])

    y_legend_offset = -.25  # + (0. if glimpses is None else -.05)
    ax.legend(bbox_to_anchor=(.90, y_legend_offset), loc=4, ncol=3, prop={'size': 6})
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if glimpses is None:
        plt.subplots_adjust(**layout)
    else:
        gs.update(**layout)

    def animate(i):
        handle.set_array(imgs[i])
        for k, v in bboxes.iteritems():
            boxes[k].set_bounds(v[i, 1], v[i, 0], v[i, 3], v[i, 2])

        handles = (handle,)
        if glimpses is not None:
            for ga, g in zip(glimpse_handles, glimpses.values()):
                ga.set_array(g[i])

            handles += glimpse_handles

        return handles

    interval = 1000. / fps
    anim = animation.FuncAnimation(fig, animate, init_func=None,
                                   frames=num_frames, interval=interval,
                                   blit=False)

    if save_as is not None:
        anim.save(save_as, dpi=dpi, savefig_kwargs=savefig_kwargs)
        plt.close(fig)
    else:
        return anim


def tile(imgs, glimpses, bboxes, fig_size=(1, 1), img_size=(1, 1), mode='vertical',
         save_as=None, n_rows=1, dpi=72, title=None):
    """

    :param imgs: image sequence of shape [T, H, W, C], T - number of timesteps
    :param glimpses: glimpses of the shape [T, n, h, w, C], n - number of glimpses per image
    :param bboxes: dict of {key: sequence of bboxes}, each sequence with shape [T, n, 4]
    :param fig_size: tuple of floats, relative scaling of the figure
    :param img_size: tuple of ints, relative size of the main image (compared to size of the glimpses)
    :param save_as: filepath, if present the figure is saved and closed.
    :return:
    """

    if not isinstance(bboxes, dict):
        bboxes = {'bbox': bboxes}

    if imgs.ndim < glimpses.ndim:
        s = glimpses.shape[:2]
        for k, b in bboxes.iteritems():
            assert s == b.shape[:2], 'Shape of "{}" doesn\t match!'.format(k)

        n_objects = glimpses.shape[1]
    else:
        glimpses = glimpses[:, np.newaxis]
        for k, v in bboxes.iteritems():
            bboxes[k] = v[:, np.newaxis]
        n_objects = 1

    colors = 'rgbc'
    # extract y, x, h, w
    for k in bboxes.keys():
        bboxes[k] = [bboxes[k][..., i] for i in xrange(4)]

    if mode == 'vertical':
        fig, axes = _tile_vertical(imgs, glimpses, bboxes, n_objects, fig_size, img_size, colors)
    elif mode == 'horizontal':
        fig, axes = _tile_horizontal(imgs, glimpses, bboxes, n_objects, fig_size, img_size, colors, n_rows)
    else:
        raise ValueError

    for ax in axes.flatten():
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    if save_as is not None:
        fig.savefig(save_as, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def _tile_vertical(imgs, glimpses, boxes, n_objects, fig_size, img_size, colors):
    # prepare figure
    yy, xx = imgs.shape[0], 1 + n_objects
    fig_y, fig_x = fig_size
    img_y, img_x = img_size

    sy, sx = yy * img_y, n_objects + img_x
    gs = gridspec.GridSpec(sy, sx)
    fig = plt.figure(figsize=(sx * fig_x, sy * fig_y))

    axes = np.empty((yy, xx), dtype=object)
    ii = 0
    for i in xrange(yy):
        axes[i, 0] = plt.subplot(gs[i * img_y:(i + 1) * img_y, :img_x])

    for i in xrange(yy):
        for j in xrange(1, xx):
            axes[i, j] = plt.subplot(gs[i * img_y:(i + 1) * img_y, j + img_x - 1])

    # plot
    for r in xrange(yy):
        axes[r, 0].imshow(imgs[r], 'gray')

        for n in xrange(n_objects):
            for (k, v), color in izip(boxes.iteritems(), colors):
                y, x, h, w = boxes[k]
                bbox = Rectangle((x[r, n], y[r, n]), w[r, n], h[r, n],
                                 edgecolor=color, facecolor='none', label=k)
                axes[r, 0].add_patch(bbox)

        for c in xrange(1, xx):
            axes[r, c].imshow(glimpses[r, c - 1], 'gray')

    # TODO: improve
    len_bbox = len(boxes)
    if len_bbox > 1:
        x_offset = .25 * len_bbox
        axes[-1, 0].legend(bbox_to_anchor=(x_offset, -.75),
                           ncol=len_bbox, loc='lower center')

    return fig, axes


def _tile_horizontal(imgs, glimpses, boxes, n_objects, fig_size, img_size, colors, n_rows):
    nt = imgs.shape[0]
    size = n_rows, nt // n_rows + int(nt % n_rows != 0)
    n_rows = 1 + n_objects
    yy, xx = size[0] * n_rows, size[1]
    fig_y, fig_x = fig_size
    img_y, img_x = img_size

    sy, sx = size[0] * (n_objects + img_y), xx * img_x
    gs = gridspec.GridSpec(sy, sx)
    fig = plt.figure(figsize=(sx * fig_x, sy * fig_y))

    axes = np.empty((yy, xx), dtype=object)

    ii = 0
    for i in xrange(yy):
        if i % n_rows == 0:
            for j in xrange(xx):
                axes[i, j] = plt.subplot(gs[ii:ii + img_y, j * img_x:(j + 1) * img_x])
            ii += img_y
        else:
            for j in xrange(xx):
                axes[i, j] = plt.subplot(gs[ii, j * img_x + img_x // 2])
            ii += 1

    for r in xrange(0, yy, n_rows):
        for c in xrange(xx):
            idx = (r // n_rows) * xx + c
            if idx < nt:
                axes[r, c].imshow(imgs[idx], 'gray')

                for n in xrange(n_objects):
                    for (k, v), color in izip(boxes.iteritems(), colors):
                        y, x, h, w = boxes[k]
                        bbox = Rectangle((x[idx, n], y[idx, n]), w[idx, n], h[idx, n],
                                         edgecolor=color, facecolor='none', label=k)
                        axes[r, c].add_patch(bbox)

                    axes[r + 1 + n, c].imshow(glimpses[idx, n], 'gray')

    len_bbox = len(boxes)
    if len_bbox > 1:
        x_offset = .25 * len_bbox
        axes[-2, axes.shape[1] // 2].legend(bbox_to_anchor=(x_offset, -(img_y + 1)),
                                            ncol=len_bbox, loc='lower center')

    return fig, axes
