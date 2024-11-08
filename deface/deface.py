#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Tuple, List, Any
from ast import literal_eval

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2

from deface import __version__
from deface.centerface import CenterFace


class ThresholdTimeline:
    def __init__(self, thresholds_by_sec, default_threshold, fps):
        self.thresholds = {}
        self.default_threshold = default_threshold
        self.fps = fps

        # Populate the thresholds dict with frame indices
        if 0 not in thresholds_by_sec:
            self.thresholds[0] = default_threshold

        for start_time, thresh in thresholds_by_sec.items():
            frame_idx = int(round(fps * start_time))
            self.thresholds[frame_idx] = thresh

    def threshold_for_frame(self, frame_idx):
        # Default to the initial threshold if no change has occurred
        threshold = self.default_threshold
        for idx, thresh in self.thresholds.items():
            if frame_idx >= idx:
                threshold = thresh
            else:
                break
        return threshold


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass
    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def has_overlap(det, other):
    x1, y1, x2, y2, _ = det
    X1, Y1, X2, Y2, _ = other
    h_overlaps = (x1 <= X2) and (x2 >= X1)
    v_overlaps = (y1 <= Y2) and (y2 >= Y1)
    return h_overlaps and v_overlaps


def has_overlap_with_union(det, union):
    return any(has_overlap(det, other) for other in union)


def unionize_overlapping_dets(dets):
    ordered_dets = sorted(dets, key=lambda x: (x[0], x[1]))
    unions = [[ordered_dets[0]]]
    # add to unions the det that have overlap with others
    for det in ordered_dets[1:]:
        if has_overlap_with_union(det, unions[-1]):
            unions[-1].append(det)
        else:
            unions.append([det])
    return unions


def get_union_rep(unions):
    union_reps = []
    # The representative of a union has centroid of weighted average centroids (by area) and
    # and the width, height and score are the max of all detections in the union
    for union in unions:
        if not union:
            continue
        elif len(union) == 1:
            union_reps.append(union[0])
        else:
            x1s = [x[0] for x in union]
            y1s = [x[1] for x in union]
            x2s = [x[2] for x in union]
            y2s = [x[3] for x in union]

            widths = [x2 - x1 for x2, x1 in zip(x2s, x1s)]
            heights = [y2 - y1 for y2, y1 in zip(y2s, y1s)]

            areas = [w * h for w, h in zip(widths, heights)]
            max_w = max(widths)
            max_h = max(heights)

            x_centroids = [np.rint((x1 + x2) / 2) for x1, x2 in zip(x1s, x2s)]
            y_centroids = [np.rint((y1 + y2) / 2) for y1, y2 in zip(y1s, y2s)]
            union_x_centroid = np.average(x_centroids, weights=areas)
            union_y_centroid = np.average(y_centroids, weights=areas)

            union_x1 = max([min(x1s), np.floor(union_x_centroid - max_w / 2)])
            union_y1 = max([min(y1s), np.floor(union_y_centroid - max_h / 2)])
            union_x2 = min([max(x2s), np.floor(union_x_centroid + max_w / 2)])
            union_y2 = min([max(y2s), np.floor(union_y_centroid + max_h / 2)])
            union_score = max([x[4] for x in union])

            union_reps.append([union_x1, union_y1, union_x2, union_y2, union_score])
    return np.asarray(union_reps, dtype=np.float32)


def filter_by_dets_history(dets, history, consistency_threshold):
    # Using a history of previous detections to assert the reliability of the new detections
    # If a new detection consistently overlap previous detections consistency_threshold times,
    #  it means the detection is reliable.
    # Add any new detection to the history
    if dets.any():
        reliables = []
        for det in dets:
            overlap_counter = 0
            for generation in history:
                for union in generation:
                    if has_overlap_with_union(det, union):
                        overlap_counter += 1
                        break
            if overlap_counter >= consistency_threshold:
                reliables.append(det)

        # Always add unionized new detections to history
        new_unions = unionize_overlapping_dets(dets)
        history.append(new_unions)
        if reliables:
            # Create unions of reliable detections
            reliable_unions = unionize_overlapping_dets(reliables)
            # Get the weighted average centeroid and max w,h and create a representative rectangle per union from those numbers.
            rep_dets = get_union_rep(reliable_unions)
            return rep_dets, history
    return np.array([]), history


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg = None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        thresholds_by_sec: Dict[float, float] = {},
        consistency_threshold: int = 2,
):
    reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader
    try:
        if 'fps' in ffmpeg_config:
            reader = imageio.get_reader(ipath, fps=ffmpeg_config['fps'])
        else:
            reader = imageio.get_reader(ipath)

        meta = reader.get_meta_data()
        _ = meta['size']
        fps = meta['fps']
    except:
        if cam:
            print(f'Could not find video device {ipath}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio. Skipping file...')
        return

    if cam:
        nframes = None
        read_iter = cam_read_iter(reader)
    else:
        read_iter = reader.iter_data()
        nframes = reader.count_frames()
    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes)

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        #  If fps is not explicitly set in ffmpeg_config, use source video fps value
        _ffmpeg_config.setdefault('fps', meta['fps'])
        # Carry over audio from input path, use "copy" codec (no transcoding) by default
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
            opath, format='FFMPEG', mode='I', **_ffmpeg_config
        )


    thresholds_timeline = ThresholdTimeline(thresholds_by_sec, threshold, fps)
    detections_history: List[Any] = []
    for frame_idx, frame in enumerate(read_iter):
        current_threshold = thresholds_timeline.threshold_for_frame(frame_idx)
        # Perform network inference, get bb dets but discard landmark predictions
        dets, _ = centerface(frame, threshold=current_threshold)

        # Use cache of the last 5 frames to get reliable detections
        reliable_dets, detections_history = filter_by_dets_history(
            dets, detections_history[-5:], consistency_threshold
        )

        # Annonymize the detections that are reliable
        anonymize_frame(
            reliable_dets, frame, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
            replaceimg=replaceimg, mosaicsize=mosaicsize
        )

        if opath is not None:
            writer.append_data(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
                cv2.destroyAllWindows()
                break
        bar.update()
    reader.close()
    if opath is not None:
        writer.close()
    bar.close()


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    # print(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    parser.add_argument(
        '--thresh', '-t', default=0.2, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument(
        "--thresholds_by_sec",
        "-tbs",
        default="{}",
        metavar="TBS",
        help="A json string (dictionary) of desired threshold by seconds (for videos only). i.e: '{1: 0.5, 5: 0.7}'",
    )
    parser.add_argument(
        "--consistency_threshold",
        "-ct",
        default=2,
        type=int,
        metavar="CT",
        choices=[0, 1, 2, 3, 4, 5],
        help="The number of previous frames (videos only) the same bounding box has to appear in to consider reliable detection. Default : 2.",
    )
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    thresholds_by_sec = literal_eval(args.thresholds_by_sec)
    consistency_threshold = args.consistency_threshold

    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                thresholds_by_sec=thresholds_by_sec,
                consistency_threshold=consistency_threshold,
            )
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
