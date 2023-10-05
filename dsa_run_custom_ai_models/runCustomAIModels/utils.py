import logging
import os
import urllib

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
import large_image
import numpy as np
import requests
from histomicstk.cli import utils as cli_utils

logging.basicConfig(level=logging.CRITICAL)


def read_input_image(args):
    # Read input image and check if it is WSI

    ts = large_image.getTileSource(args.inputImageFile, style=args.style)

    ts_metadata = ts.getMetadata()

    is_wsi = ts_metadata['magnification'] is not None

    return ts, is_wsi


def image_inversion_flag_setter(args=None):
    # Generates image inversion flags
    invert_image, default_img_inversion = False, False
    if args.ImageInversionForm == 'Yes':
        invert_image = True
    if args.ImageInversionForm == 'No':
        invert_image = False
    if args.ImageInversionForm == 'default':
        default_img_inversion = True
    return invert_image, default_img_inversion


def validate_args(args):
    # Validates the input arguments
    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

    if len(args.reference_mu_lab) != 3:
        raise ValueError('Reference Mean LAB should be a 3 element vector.')

    if len(args.reference_std_lab) != 3:
        raise ValueError('Reference Stddev LAB should be a 3 element vector.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')


def process_wsi_as_whole_image(
        ts, invert_image=False, args=None, default_img_inversion=False):

    # Segment wsi foreground at low resolution
    im_fgnd_mask_lres, fgnd_seg_scale = \
        cli_utils.segment_wsi_foreground_at_low_res(
            ts, invert_image=invert_image, frame=args.frame,
            default_img_inversion=default_img_inversion)
    return im_fgnd_mask_lres, fgnd_seg_scale


def process_wsi(ts, it_kwargs, args, im_fgnd_mask_lres=None,
                fgnd_seg_scale=None, process_whole_image=False):

    num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

    if process_whole_image:

        tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
            args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
            it_kwargs, style=args.style
        )

    else:

        tile_fgnd_frac_list = np.full(num_tiles, 1.0)

    num_fgnd_tiles = np.count_nonzero(
        tile_fgnd_frac_list >= args.min_fgnd_frac)

    if not num_fgnd_tiles:
        tile_fgnd_frac_list = np.full(num_tiles, 1.0)
        num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list)

    return tile_fgnd_frac_list


def compute_reinhard_norm(args, invert_image=False,
                          default_img_inversion=False):

    src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats(
        args.inputImageFile, 0.01, magnification=args.analysis_mag,
        invert_image=invert_image, style=args.style, frame=args.frame,
        default_img_inversion=default_img_inversion)
    return src_mu_lab, src_sigma_lab


def generate_mask(im_tile, args, src_mu_lab, src_sigma_lab):
    # Flags
    single_channel = False
    invert_image = False

    # Get tile image & check number of channels
    single_channel = len(
        im_tile['tile'].shape) <= 2 or im_tile['tile'].shape[2] == 1
    if single_channel:
        im_tile = np.dstack(
            (im_tile['tile'], im_tile['tile'], im_tile['tile']))
        if args.ImageInversionForm == 'Yes':
            invert_image = True
    else:
        im_tile = im_tile['tile'][:, :, :3]

    # Perform image inversion
    if invert_image:
        im_tile = np.max(im_tile) - im_tile

    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 args.reference_mu_lab,
                                 args.reference_std_lab,
                                 src_mu=src_mu_lab,
                                 src_sigma=src_sigma_lab)

    # Perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    # Perform deconvolution
    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains
    im_nuclei_stain = im_stains[:, :, 0].astype(float)

    # Segment nuclear foreground
    im_nuclei_fgnd_mask = im_nuclei_stain < args.foreground_threshold

    # Segment nuclei
    im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
        im_nuclei_stain,
        im_nuclei_fgnd_mask,
        args.min_radius,
        args.max_radius,
        args.min_nucleus_area,
        args.local_max_search_radius
    )
    return im_nuclei_seg_mask


def detect_nuclei_with_ai(ts, tile_fgnd_frac_list, it_kwargs, args,
                          invert_image=False, is_wsi=False, default_img_inversion=False,
                          nuclei_center_coordinates=False, process_whole_image=False):

    modelList = {
        'Nuclick Classification': 'nuclick_classification',
        'Nuclick Segmentation': 'nuclick_segmentation',
        'Segment Anything': 'segment_anything',
        'Segment Anything onclick': 'segment_anything_onclick',
        'Mobile Segment Anything': 'segment_anything_mobile',
    }
    network_location = 'http://localhost:8000' if not args.ai_model else args.ai_model
    if not urllib.parse.urlparse(network_location).path and args.prebuild_ai_models in modelList:
        network_location = (network_location.rstrip('/') + '/' +
                            modelList[args.prebuild_ai_models] + '/')

    tile_nuclei_list = []

    tile_nuclei_class = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        # Sub process for mask and annotations
        if args.send_mask_tiles or args.send_nuclei_annotations:
            if is_wsi:
                if process_whole_image:
                    im_fgnd_mask_lres, fgnd_seg_scale = process_wsi_as_whole_image(
                        ts, invert_image=invert_image, args=args,
                        default_img_inversion=default_img_inversion)
                    tile_fgnd_frac_list = process_wsi(ts,
                                                            it_kwargs,
                                                            args,
                                                            im_fgnd_mask_lres,
                                                            fgnd_seg_scale,
                                                            process_whole_image)
                else:
                    tile_fgnd_frac_list = process_wsi(ts, it_kwargs, args)
        # Compute reinhard stats for color normalization
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
                    src_mu_lab, src_sigma_lab = compute_reinhard_norm(
                        args, invert_image=invert_image, default_img_inversion=default_img_inversion)
                    pass

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # Extract tile information.
        gx, gy, gh, gw, x, y = tile['gx'], tile['gy'], tile['gheight'], tile[
            'gwidth'], tile['x'], tile['y']

        # Prepare payload for HTTP request.
        payload = {"model_type": args.type_ai_models}

        # Include nuclei center in payload if specified
        if args.nuclei_center:
            nuclei_locations = []
            for i in range(0, len(args.nuclei_center), 2):
                nuclei_locations.append(
                    [args.nuclei_center[i], args.nuclei_center[i + 1]])
            payload["nuclei_location"] = nuclei_locations

        # Include image data in payload if specified.
        if args.send_image_tiles:
            payload["image"] = np.asarray(tile['tile'][:, :, :3]).tolist()

        # Include mask data in payload if specified.
        if args.send_mask_tiles:
            # Generate nuclei mask.
            nuclei_mask = generate_mask(tile, args, src_mu_lab, src_sigma_lab)
            payload["mask"] = np.asarray(nuclei_mask).tolist()

        # Include nuclei annotations in payload if specified.
        if args.send_nuclei_annotations:
            cur_nuclei_list = htk_nuclear.detect_tile_nuclei(
                tile,
                args,
                src_mu_lab, src_sigma_lab, invert_image=invert_image,
                default_img_inversion=default_img_inversion,
            )
            payload["nuclei"] = cur_nuclei_list

        # Include tile size information in payload.
        payload["tilesize"] = (gx, gy, gh, gw, x, y)

        try:
            # Send the HTTP POST request.
            response = requests.post(network_location, json=payload)

            if response.status_code == 200:
                # Handle response data if successful.
                output = response.json()
                if "network_output" in output:
                    tile_nuclei_list.append(
                        response.json().get("network_output"))
            else:
                # Handle request failure.
                print(
                    f"Request failed with status code: {response.status_code}")
                tile_nuclei_class.append([])
        except requests.exceptions.RequestException as e:
            # Handle request exception.
            print(f"Request error: {e}")
        except Exception as e:
            # Handle other exceptions.
            print(f"Error: {e}")

        # Flatten the list of nuclei annotations.
        nuclei_list = [
            anot for anot_list in tile_nuclei_list for anot in anot_list]

        if nuclei_center_coordinates and args.type_ai_models == "Segmentation":
            break
    return nuclei_list
