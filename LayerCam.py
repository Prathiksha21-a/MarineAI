"""Object-centric LayerCAM explanations for the MarineAI YOLO11 fish detector.

Public API:
    generate_gradcam(input_path, output_path)

Each detected fish is explained independently in a padded crop. This avoids
trying to match a post-NMS detection to an unreliable raw YOLO candidate.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import LayerCAM
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from detector import FISH_CONF, fish_model


MODEL_PATH = "models/fish_best.pt"
CAM_IMAGE_SIZE = 640
CROP_CONTEXT = 0.08
_cam_model = None


class _YOLOForward(torch.nn.Module):
    """Expose the raw YOLO output tensor required by pytorch-grad-cam."""

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, image):
        output = self.yolo_model(image)
        return output[0] if isinstance(output, (tuple, list)) else output


class _FishClassTarget:
    """Target the strongest fish-class response in an object-specific crop."""

    def __init__(self, class_id):
        self.class_id = class_id

    def __call__(self, output):
        # BaseCAM calls targets with one batch item: [channels, anchors].
        scores = (
            output[4 + self.class_id]
            if output.ndim == 2
            else output[0, 4 + self.class_id]
        )
        return scores.max()


def _get_cam_model():
    """Load a YOLO model used only for autograd-based explanations."""
    global _cam_model

    if _cam_model is None:
        _cam_model = YOLO(MODEL_PATH).model.eval()

        for parameter in _cam_model.parameters():
            parameter.requires_grad_(True)

    return _cam_model


def _get_target_layers(yolo_model):
    """Return the Detect-head's own input layers (P3/P4/..), finest first.

    These are the last per-scale feature maps computed by the neck before
    the Detect head reads them. P3 (the first entry) has the smallest
    stride (highest spatial resolution, e.g. 80x80 for a 640 input), so a
    CAM taken here traces the fish's silhouette.
    """
    layers = yolo_model.model
    detect_index = len(layers) - 1
    detect_head = layers[detect_index]

    if not hasattr(detect_head, "f"):
        raise ValueError("The YOLO model does not expose Detect-head inputs.")

    detect_inputs = detect_head.f
    if isinstance(detect_inputs, int):
        detect_inputs = [detect_inputs]

    def resolve_source(source):
        return detect_index + source if source < 0 else source

    resolved_inputs = [resolve_source(source) for source in detect_inputs]

    # detect_inputs are ordered P3 -> P4 -> P5 (fine -> coarse) in
    # YOLOv8/11, so this list is already finest-resolution first.
    return [layers[index] for index in resolved_inputs]


def _letterbox_tensor(rgb_image):
    """Letterbox an RGB crop and return tensor plus placement metadata."""
    height, width = rgb_image.shape[:2]
    gain = min(CAM_IMAGE_SIZE / height, CAM_IMAGE_SIZE / width)

    resized_width = round(width * gain)
    resized_height = round(height * gain)

    left = round((CAM_IMAGE_SIZE - resized_width) / 2 - 0.1)
    top = round((CAM_IMAGE_SIZE - resized_height) / 2 - 0.1)

    letterbox = LetterBox(
        new_shape=(CAM_IMAGE_SIZE, CAM_IMAGE_SIZE),
        auto=False,
        stride=32,
    )

    padded = letterbox(image=rgb_image)
    tensor = torch.from_numpy(
        padded.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return tensor / 255.0, (resized_width, resized_height, left, top)


def _padded_crop(image, box_xyxy):
    """Crop one detected fish with a small amount of surrounding context."""
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)

    padding = CROP_CONTEXT * max(x2 - x1, y2 - y1)

    crop_x1 = max(0, int(np.floor(x1 - padding)))
    crop_y1 = max(0, int(np.floor(y1 - padding)))
    crop_x2 = min(image_width, int(np.ceil(x2 + padding)))
    crop_y2 = min(image_height, int(np.ceil(y2 + padding)))

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def _local_box(crop_bounds, fish_box):
    """Fish box coordinates translated into the crop's local pixel space."""
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    fish_x1, fish_y1, fish_x2, fish_y2 = np.asarray(fish_box, dtype=np.int32)

    fish_x1 = max(crop_x1, fish_x1)
    fish_y1 = max(crop_y1, fish_y1)
    fish_x2 = min(crop_x2, fish_x2)
    fish_y2 = min(crop_y2, fish_y2)

    if fish_x2 <= fish_x1 or fish_y2 <= fish_y1:
        return None

    return (
        fish_x1 - crop_x1,
        fish_y1 - crop_y1,
        fish_x2 - crop_x1,
        fish_y2 - crop_y1,
    )


def _fill_mask_holes(mask):
    """Fill background pockets fully enclosed by foreground (imfill)."""
    height, width = mask.shape
    flood_source = mask.copy()
    flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood_source, flood_fill_mask, (0, 0), 1)

    unreachable_background = 1 - flood_source
    return np.maximum(mask, unreachable_background).astype(np.uint8)


def _largest_component(mask, proximity):
    """Keep the main foreground blob, plus any nearby disconnected blob.

    A thin, low-contrast tail fin often gets colour-classified as
    background right where it narrows to meet the body (the caudal
    peduncle), which severs it from the main blob in the mask -- not
    because it isn't fish, but because thin/translucent structures are a
    known weak point of colour-based segmentation like GrabCut. The
    previous version kept only the single largest component, which
    silently discarded that tail fragment.

    Instead: keep the largest component, then also keep any other
    component that falls within `proximity` pixels of it (dilate the main
    blob and test for overlap). A stray blob far from the fish's body --
    real background clutter, a second fish only partly in this crop's
    padding -- is still discarded, since it won't fall inside that
    dilated reach. This only reconnects fragments that are spatially
    plausible parts of the same fish.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_component = (labels == largest_label).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (proximity * 2 + 1, proximity * 2 + 1)
    )
    reach = cv2.dilate(main_component, kernel).astype(bool)

    keep = main_component.copy()
    for label in range(1, num_labels):
        if label == largest_label:
            continue
        component = labels == label
        if np.any(reach & component):
            keep[component] = 1

    return keep


def _clean_silhouette(mask, local_box):
    """Morphologically clean a raw GrabCut mask into hard + soft versions.

    Returns:
        hard_mask: binary mask (holes filled, speckle removed). This is the
            authority on "is this pixel part of the fish" -- used both for
            CAM statistics and, now, directly as the alpha/coverage signal.
        soft_mask: hard_mask with a ~1px feathered edge, applied only at the
            very final blend so silhouette boundaries anti-alias instead of
            showing a jagged pixel staircase.
    """
    x1, y1, x2, y2 = local_box
    box_extent = max(1, min(x2 - x1, y2 - y1))
    kernel_size = max(5, int(round(box_extent * 0.06)) | 1)  # odd, >=5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    filled = _fill_mask_holes(closed)

    proximity = max(5, int(round(box_extent * 0.10)))
    hard_mask = _largest_component(filled, proximity)
    # A component reconnected via proximity can itself introduce a new
    # small gap where it meets the main blob -- run hole-filling once more
    # now that the fragments are unified.
    hard_mask = _fill_mask_holes(hard_mask)

    soft_mask = cv2.GaussianBlur(
        hard_mask.astype(np.float32), (0, 0), sigmaX=1.0
    )
    return hard_mask, soft_mask


def _ellipse_fallback_mask(shape, local_box):
    """Elliptical footprint inscribed in the box, used if GrabCut degenerates."""
    x1, y1, x2, y2 = local_box
    mask = np.zeros(shape, dtype=np.uint8)

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, thickness=-1)
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.0)
    return mask, soft_mask


def _refine_with_second_grabcut(crop_rgb, raw_silhouette):
    """Re-run GrabCut seeded with the first pass's own shape as a prior."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(raw_silhouette, kernel, iterations=2)
    probable_region = cv2.dilate(raw_silhouette, kernel, iterations=3)

    mask = np.full(crop_rgb.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[probable_region.astype(bool)] = cv2.GC_PR_BGD
    mask[raw_silhouette.astype(bool)] = cv2.GC_PR_FGD
    mask[sure_fg.astype(bool)] = cv2.GC_FGD

    if not sure_fg.any():
        return raw_silhouette

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            np.ascontiguousarray(crop_rgb),
            mask,
            None,
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return raw_silhouette

    refined = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    if refined.sum() < 0.5 * raw_silhouette.sum():
        return raw_silhouette

    return refined


def _fish_silhouette(crop_rgb, local_box):
    """Create a conservative fish mask inside the YOLO detection box.

    The YOLO bounding box is treated as the absolute maximum region
    where the LayerCAM explanation is allowed to appear.

    GrabCut is initialized with a smaller foreground seed so that
    surrounding blue water is less likely to become foreground.
    """

    x1, y1, x2, y2 = [int(v) for v in local_box]

    height, width = crop_rgb.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    if box_area < 400:
        return _ellipse_fallback_mask(
            crop_rgb.shape[:2],
            (x1, y1, x2, y2)
        )

    mask = np.full(
        (height, width),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    margin_x = max(2, int(box_width * 0.15))
    margin_y = max(2, int(box_height * 0.15))

    inner_x1 = min(x2 - 1, x1 + margin_x)
    inner_y1 = min(y2 - 1, y1 + margin_y)
    inner_x2 = max(inner_x1 + 1, x2 - margin_x)
    inner_y2 = max(inner_y1 + 1, y2 - margin_y)

    mask[
        inner_y1:inner_y2,
        inner_x1:inner_x2
    ] = cv2.GC_PR_FGD

    center_x = (inner_x1 + inner_x2) // 2
    center_y = (inner_y1 + inner_y2) // 2

    axes_x = max(
        2,
        int((inner_x2 - inner_x1) * 0.30)
    )

    axes_y = max(
        2,
        int((inner_y2 - inner_y1) * 0.30)
    )

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (axes_x, axes_y),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1
    )

    bgd_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    try:
        cv2.grabCut(
            np.ascontiguousarray(crop_rgb),
            mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

    except cv2.error:
        return _ellipse_fallback_mask(
            crop_rgb.shape[:2],
            (x1, y1, x2, y2)
        )

    raw_silhouette = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype(np.uint8)

    box_mask = np.zeros_like(
        raw_silhouette,
        dtype=np.uint8
    )

    box_mask[y1:y2, x1:x2] = 1

    raw_silhouette = (
        raw_silhouette * box_mask
    ).astype(np.uint8)

    inside_area = raw_silhouette[
        y1:y2,
        x1:x2
    ].sum()

    if inside_area < 0.10 * box_area:
        return _ellipse_fallback_mask(
            crop_rgb.shape[:2],
            (x1, y1, x2, y2)
        )

    refined_silhouette = _refine_with_second_grabcut(
        crop_rgb,
        raw_silhouette
    )

    refined_silhouette = (
        refined_silhouette * box_mask
    ).astype(np.uint8)

    hard_mask, soft_mask = _clean_silhouette(
        refined_silhouette,
        (x1, y1, x2, y2)
    )

    hard_mask = (
        hard_mask * box_mask
    ).astype(np.uint8)

    soft_mask = (
        soft_mask * box_mask.astype(np.float32)
    )

    return hard_mask, soft_mask


def _normalise_object_cam(cam, hard_mask):
    """Rescale CAM within the fish silhouette to the full 0-1 range.

    - Bilateral filtering smooths the CAM while respecting the underlying
      image's edges (won't blur activation across the fish's own outline).
    - Min-max stretch uses only hard_mask pixel values, so background never
      skews the contrast range.
    - Gamma < 1 lifts real-but-weaker mid activation so the colour ramp
      isn't dominated by one hotspot.
    - IMPORTANT: unlike the previous version, nothing here is thresholded
      to zero. A pixel with genuinely low (but real) activation stays a
      real, low, positive number -- it will render as cool blue rather
      than vanishing. Whether a pixel is drawn on at all is now a separate
      decision made in _overlay_cam, driven purely by the silhouette.
    """
    cam = cv2.bilateralFilter(cam.astype(np.float32), d=7, sigmaColor=0.15, sigmaSpace=3)

    hard_bool = hard_mask.astype(bool)
    silhouette_values = cam[hard_bool]
    if silhouette_values.size == 0:
        return np.zeros_like(cam, dtype=np.float32)

    low = float(np.percentile(silhouette_values, 2.0))
    high = float(np.percentile(silhouette_values, 98.0))

    if high - low <= 1e-8:
        cam = np.full_like(cam, 0.5, dtype=np.float32)
    else:
        cam = np.clip((cam - low) / (high - low), 0.0, 1.0)

    cam = cam ** 0.6  # gamma < 1 lifts mid activation for fuller coverage
    cam[~hard_bool] = 0.0
    return cam


def _single_layer_cam(cam_runner, target_layer, input_tensor, class_id):
    """Raw CAM (still letterbox-padded) from exactly one target layer."""
    with LayerCAM(model=cam_runner, target_layers=[target_layer]) as layer_cam:
        return layer_cam(
            input_tensor=input_tensor,
            targets=[_FishClassTarget(class_id)],
        )[0]


def _cam_for_crop(cam_runner, target_layers, crop_rgb, class_id, device, local_box):
    """Generate silhouette-masked CAM intensity + coverage for a fish crop.

    Returns (crop_cam, crop_coverage):
        crop_cam: 0-1 intensity map (what colour each pixel gets). Zero
            outside the silhouette, otherwise always a real value -- never
            hard-thresholded to zero inside the fish.
        crop_coverage: soft 0-1 mask (what gets colour drawn at all, and
            how strongly, independent of intensity). This is what fixes
            "gray patches inside the fish": coverage is decided purely by
            the GrabCut silhouette, not by how strong the activation is.
    """
    input_tensor, (resized_width, resized_height, left, top) = (
        _letterbox_tensor(crop_rgb)
    )
    input_tensor = input_tensor.to(device)

    fused_padded_cam = None
    for target_layer in target_layers:
        layer_cam = _single_layer_cam(cam_runner, target_layer, input_tensor, class_id)
        fused_padded_cam = (
            layer_cam
            if fused_padded_cam is None
            else np.maximum(fused_padded_cam, layer_cam)
        )

    crop_cam = fused_padded_cam[
        top:top + resized_height,
        left:left + resized_width,
    ]

    crop_cam = cv2.resize(
        crop_cam,
        (crop_rgb.shape[1], crop_rgb.shape[0]),
    )

    hard_mask, soft_mask = _fish_silhouette(crop_rgb, local_box)
    crop_cam = _normalise_object_cam(crop_cam, hard_mask)
    return crop_cam, hard_mask.astype(np.float32)


def _merge_into_full(full_map, crop_map, crop_bounds, fish_box):
    """Place a per-crop map into the full image, clipped to its box.

    Shared by both the intensity merge and the coverage merge below -- same
    box-clipping safety net either way, just applied to two different maps.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    fish_x1, fish_y1, fish_x2, fish_y2 = np.asarray(
        fish_box,
        dtype=np.int32,
    )

    fish_x1 = max(crop_x1, fish_x1)
    fish_y1 = max(crop_y1, fish_y1)
    fish_x2 = min(crop_x2, fish_x2)
    fish_y2 = min(crop_y2, fish_y2)

    if fish_x2 <= fish_x1 or fish_y2 <= fish_y1:
        return

    local_x1 = fish_x1 - crop_x1
    local_y1 = fish_y1 - crop_y1
    local_x2 = fish_x2 - crop_x1
    local_y2 = fish_y2 - crop_y1

    crop_mask = np.zeros_like(crop_map, dtype=np.float32)
    crop_mask[local_y1:local_y2, local_x1:local_x2] = crop_map[
        local_y1:local_y2,
        local_x1:local_x2,
    ]

    full_map[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(
        full_map[crop_y1:crop_y2, crop_x1:crop_x2],
        crop_mask,
    )


def _overlay_cam(rgb_image, merged_cam, merged_coverage):
    """Render the CAM, colouring exactly the fish silhouettes -- fully.

    alpha now comes from merged_coverage (the silhouette) alone, NOT from
    whether merged_cam is above some cutoff. That's the actual fix: every
    silhouette pixel gets painted, coloured by its true (possibly low)
    CAM value -- low activation shows as cool blue, not as a see-through
    gap revealing the raw photo underneath.
    """
    heatmap = cv2.applyColorMap(
        np.uint8(merged_cam * 255),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    strict_mask = (merged_coverage > 0.5).astype(np.float32)

    merged_cam = merged_cam * strict_mask
    alpha = strict_mask[..., None] * 0.55

    return (
        rgb_image * (1.0 - alpha) + heatmap * alpha
    ).astype(np.uint8)


def generate_gradcam(input_path, output_path):
    """Save a merged object-centric LayerCAM visualization for every fish.

    The existing detector model supplies final boxes. Each box is explained in
    its own padded crop and all box-limited CAMs are merged into one image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr_image = cv2.imread(str(input_path))
    if bgr_image is None:
        raise ValueError(f"Could not read image: {input_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    detections = fish_model.predict(
        str(input_path),
        conf=FISH_CONF,
        verbose=False,
    )[0]

    if len(detections.boxes) == 0:
        cv2.imwrite(str(output_path), bgr_image)
        return False

    boxes = detections.boxes.xyxy.detach().cpu().numpy().copy()
    class_ids = detections.boxes.cls.detach().cpu().numpy().astype(int)

    yolo_model = _get_cam_model()
    device = next(yolo_model.parameters()).device

    cam_runner = _YOLOForward(yolo_model).to(device).eval()

    # Use only the finest 1-2 Detect-head input scales (P3, optionally P4).
    # P5 (stride 32) is left out entirely: at crop resolution it degrades
    # to a few giant cells and drags the merged CAM back towards a blocky
    # box-shaped blob.
    target_layers = _get_target_layers(yolo_model)[:1]

    merged_cam = np.zeros(rgb_image.shape[:2], dtype=np.float32)
    merged_coverage = np.zeros(rgb_image.shape[:2], dtype=np.float32)

    for fish_box, class_id in zip(boxes, class_ids):
        crop_rgb, crop_bounds = _padded_crop(rgb_image, fish_box)

        if crop_rgb.size == 0:
            continue

        local_box = _local_box(crop_bounds, fish_box)
        if local_box is None:
            continue

        crop_cam, crop_coverage = _cam_for_crop(
            cam_runner,
            target_layers,
            crop_rgb,
            int(class_id),
            device,
            local_box,
        )

        _merge_into_full(merged_cam, crop_cam, crop_bounds, fish_box)
        _merge_into_full(merged_coverage, crop_coverage, crop_bounds, fish_box)

    visualization = _overlay_cam(rgb_image, merged_cam, merged_coverage)

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
    )

    return True