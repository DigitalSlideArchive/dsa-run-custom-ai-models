import json
import os
from pathlib import Path

import histomicstk
import histomicstk.segmentation.label as htk_seg_label
import large_image
import numpy as np
import utils
from histomicstk.cli.utils import CLIArgumentParser


def main(args):

    # Flags
    invert_image = False
    default_img_inversion = False
    process_whole_image = False
    nuclei_center_coordinates = False

    utils.validate_args(args)

    # Check if the whole slide should be analyzed
    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True

    # Check if the nuclei coordinates are present
    if np.all(np.array(args.nuclei_center) == -1):
        nuclei_center_coordinates = True

    # Provide default value for tile_overlap
    tile_overlap = args.tile_overlap_value
    if tile_overlap == -1:
        tile_overlap = (args.max_radius + 1) * 4

    # Retrive style
    if not args.style or args.style.startswith('{#control'):
        args.style = None

    # Initial arguments
    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size, 'height': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
        'tile_overlap': {'x': tile_overlap, 'y': tile_overlap},
        'style': {args.style}
    }

    # Retrive frame
    if not args.frame or args.frame.startswith('{#control'):
        args.frame = None
    elif not args.frame.isdigit():
        raise Exception("The given frame value is not an integer")
    else:
        it_kwargs['frame'] = args.frame

    #
    # Color inversion flag
    #
    invert_image, default_img_inversion = utils.image_inversion_flag_setter(
        args)

    #
    # Read Input Image
    #
    ts, is_wsi = utils.read_input_image(args)
    tile_fgnd_frac_list = [1.0]

    #
    # Automatically deciding the tile size
    #
    if process_whole_image and not nuclei_center_coordinates:

        for i in range(0, len(args.nuclei_center), 2):
            x_array = []
            y_array = []
            x_array.append(args.nuclei_center[i])
            y_array.append(args.nuclei_center[i + 1])
        x_avg = np.average(x_array)
        y_avg = np.average(y_array)
        print(f"this is the nuclei area: {args.min_nucleus_area}")

        it_kwargs['region'] = {
            'left': np.abs(x_avg - args.min_nucleus_area) if x_avg > args.min_nucleus_area else 0,
            'top': np.abs(y_avg - args.min_nucleus_area) if y_avg > args.min_nucleus_area else 0,
            'width': args.min_nucleus_area * 2,
            'height': args.min_nucleus_area * 2,
            'units': 'base_pixels'
        }
        ######################################################

    if not process_whole_image:

        it_kwargs['region'] = {
            'left': args.analysis_roi[0],
            'top': args.analysis_roi[1],
            'width': args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units': 'base_pixels'
        }

    if is_wsi:

        if process_whole_image:

            im_fgnd_mask_lres, fgnd_seg_scale = utils.process_wsi_as_whole_image(
                ts, invert_image=invert_image, args=args,
                default_img_inversion=default_img_inversion)
            tile_fgnd_frac_list = utils.process_wsi(ts,
                                                    it_kwargs,
                                                    args,
                                                    im_fgnd_mask_lres,
                                                    fgnd_seg_scale,
                                                    process_whole_image)
        else:
            tile_fgnd_frac_list = utils.process_wsi(ts, it_kwargs, args)

    #
    # Compute reinhard stats for color normalization
    #
    src_mu_lab = None
    src_sigma_lab = None

    if is_wsi and process_whole_image:
        # Get a tile
        tile_info = ts.getSingleTile(
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            frame=args.frame)
        # Get tile image & check number of channels
        single_channel = len(
            tile_info['tile'].shape) <= 2 or tile_info['tile'].shape[2] == 1
        if not single_channel:
            src_mu_lab, src_sigma_lab = utils.compute_reinhard_norm(
                args, invert_image=invert_image, default_img_inversion=default_img_inversion)

    #
    # Process nuclei using AI models
    #
    nuclei_list = utils.detect_nuclei_with_ai(
        ts,
        tile_fgnd_frac_list,
        it_kwargs,
        args,
        invert_image,
        is_wsi,
        src_mu_lab,
        src_sigma_lab,
        default_img_inversion=default_img_inversion,
        nuclei_center_coordinates=nuclei_center_coordinates)

    #
    # Remove overlapping nuclei
    #
    if args.remove_overlapping_nuclei_segmentation:

        nuclei_list = htk_seg_label.remove_overlap_nuclei(
            nuclei_list, args.nuclei_annotation_format)

    #
    # Write annotation file
    #
    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        'name': annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        'elements': nuclei_list,
        'attributes': {
            'src_mu_lab': None if src_mu_lab is None else src_mu_lab.tolist(),
            'src_sigma_lab': None if src_sigma_lab is None else src_sigma_lab.tolist(),
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,
        },
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(
            annotation,
            annotation_file,
            separators=(
                ',',
                ':'),
            sort_keys=False)


if __name__ == '__main__':

    main(CLIArgumentParser().parse_args())
