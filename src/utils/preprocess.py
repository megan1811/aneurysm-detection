import numpy as np
import SimpleITK as sitk
from pathlib import Path
from SimpleITK import Image as SITKImage
import pydicom

REQUIRED_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"]


def dicom_pixel_to_world(ds: pydicom.Dataset, x: float, y: float) -> np.array:
    """
    Convert DICOM pixel coordinates (x, y) into a 3D patient-space (LPS) world coordinate
    using the slice's position, orientation, and pixel spacing metadata.

    Args:
        ds: pydicom Dataset for the slice, containing `ImagePositionPatient`,
            `ImageOrientationPatient`, and `PixelSpacing`.
        x (float): Column index (pixel coordinate) in the image.
        y (float): Row index (pixel coordinate) in the image.

    Returns:
        np.array: A (3,) array with the corresponding world coordinate in millimeters.
    """
    missing = [t for t in REQUIRED_TAGS if t not in ds]
    if missing:
        raise ValueError(
            f"DICOM is missing required tags {missing}. "
            f"Cannot compute world coordinates for SOPInstanceUID={ds.SOPInstanceUID}, modality={ds.Modality}"
        )

    image_position = np.array(ds.ImagePositionPatient, dtype=np.float32)  # (x0, y0, z0)
    orientation = np.array(ds.ImageOrientationPatient, dtype=np.float32)  # (6,)
    pixel_spacing = np.array(ds.PixelSpacing, dtype=np.float32)  # (sx, sy)

    row_direction = orientation[:3]
    col_direction = orientation[3:]

    row_spacing, col_spacing = pixel_spacing
    world_coord = (
        image_position
        + x * col_spacing * col_direction
        + y * row_spacing * row_direction
    )

    return world_coord


def load_dicom_series(dicom_folder: Path) -> SITKImage:
    """
    Load a DICOM series from a folder into a SimpleITK 3D image.

    Args:
        dicom_folder (Path): Directory containing a single DICOM series.

    Returns:
        SITKImage: A 3D SimpleITK image reconstructed from the series.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise RuntimeError("No DICOM series found")
    # TODO: handle multiple series if needed
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()

    return image


def resample_image(
    image: SITKImage, new_spacing: tuple = (1, 1, 1), is_label: bool = False
) -> SITKImage:
    """
    Resample a 3D image to a new voxel spacing using linear or nearest-neighbor interpolation.

    Args:
        image (SITKImage): Input SimpleITK image.
        new_spacing (tuple): Desired output spacing in millimeters (x, y, z).
        is_label (bool): Whether the image is a label map (uses nearest-neighbor interpolation).

    Returns:
        SITKImage: The resampled image with updated spacing and size.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def reorient_to_LPS(image: SITKImage) -> SITKImage:
    """
    Canonicalize voxel axis orientation to LPS (Left–Posterior–Superior).

    IMPORTANT:
    - This does NOT change the patient coordinate system (which is already LPS
      for all DICOM images).
    - This function standardizes how voxel axes map to patient directions so that:
        * voxel +X always points toward Left
        * voxel +Y always points toward Posterior
        * voxel +Z always points toward Superior
    - This guarantees consistent spatial semantics across scans, which is critical
      when using voxel coordinates or aggregating spatial information.

    In short:
    - DICOM guarantees LPS *patient space*
    - This enforces LPS *voxel orientation*

    Args:
        image (SITKImage): Input image in arbitrary voxel orientation.

    Returns:
        SITKImage: Image with voxel axes aligned to canonical LPS orientation.
    """
    transform = sitk.DICOMOrientImageFilter()
    transform.SetDesiredCoordinateOrientation("LPS")
    return transform.Execute(image)


def normalize_intensity(image: SITKImage, modality="MRI") -> SITKImage:
    """
    Normalize image intensities based on modality, using windowing for CT or percentile
    clipping for MRI, followed by min–max scaling.

    Args:
        image (SITKImage): Input image to normalize.
        modality (str): Imaging modality ("CT", "MRI", etc.) determining normalization strategy.

    Returns:
        SITKImage: Intensity-normalized image with original metadata preserved.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if modality in ["CT", "CTA"]:
        # Windowing typical for CT (example window -100 to 700 HU)
        low = -100
        high = 700
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low + 1e-8)

    elif modality in ["MR", "MRA", "MRI", "MRI T1", "MRI T2"]:
        # Percentile clipping
        p_low, p_high = 1, 99
        low_val, high_val = np.percentile(arr, [p_low, p_high])
        arr = np.clip(arr, low_val, high_val)
        arr = (arr - low_val) / (high_val - low_val + 1e-8)
        # Optionally z-score normalization could be applied here as well

    else:
        # Fallback: min-max scaling
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    img_norm = sitk.GetImageFromArray(arr)
    img_norm.CopyInformation(image)
    return img_norm


def load_and_preprocess(series_id: str, modality: str, base_dir: Path) -> SITKImage:
    """
    Load a DICOM series and apply standard preprocessing: reorientation, resampling,
    and modality-specific intensity normalization.

    Args:
        series_id (str): Identifier for the DICOM series.
        modality (str): Imaging modality used for normalization.
        base_dir (Path): Base directory containing the DICOM series folder.

    Returns:
        SITKImage: The fully preprocessed 3D image.
    """
    series_path = base_dir / "series" / series_id
    if not series_path.exists():
        raise FileNotFoundError(f"Series path {series_path} does not exist.")

    image = load_dicom_series(series_path)
    if image.GetDimension() == 4:
        raise ValueError("4D DICOM series are not supported.")

    # Canonicalize voxel axes so that all scans share the same
    # Left/Posterior/Superior axis semantics (patient space remains LPS)
    image = reorient_to_LPS(image)

    image = resample_image(image, new_spacing=[1, 1, 1], is_label=False)
    image = normalize_intensity(image, modality=modality)

    return image


def jitter_coords(vox_coord: np.array, max_jitter: float = 8.0) -> tuple:
    """
    Add random uniform jitter to a 3D world coordinate.

    Args:
        vox_coord (np.array): Voxel coordinate of aneurism in sltk image (3 ,).
        max_jitter (float): Maximum absolute jitter in millimeters applied along each axis.

    Returns:
        tuple: Jittered voxek coordinate as a 3-element tuple of floats.
    """
    jitter = np.random.uniform(-max_jitter, max_jitter, size=3)
    return tuple([int(vox_dim) for vox_dim in np.array(vox_coord) + jitter])


def world_to_voxel(sitk_img: SITKImage, world_coord: np.array) -> tuple | None:
    """
    Convert a world-space coordinate to a voxel index, or return None if outside.

    Args:
        sitk_img (SITKImage): SimpleITK image defining the physical space.
        world_coord: 3D point in the image’s physical (LPS) coordinate system.

    Returns:
        tuple | None: Integer voxel index (x, y, z) if inside the volume, else None.
    """
    # Use continuous index so we can bounds-check ourselves.
    continuous_index = sitk_img.TransformPhysicalPointToContinuousIndex(world_coord)
    size = sitk_img.GetSize()

    for c, s in zip(continuous_index, size):
        if c < 0 or c >= s:
            return None

    return tuple(int(round(c)) for c in continuous_index)


def extract_patch_sitk(
    sitk_img: SITKImage, center_voxel: np.array, patch_size: int = 32
) -> SITKImage:
    """
    Extract a cubic 3D patch from a SimpleITK image around a given world-space center.

    Args:
        sitk_img (SITKImage): Input 3D image.
        center_vox_coord: Center voxel of desired patch.
        patch_size (int): Edge length of the cubic patch in voxels.

    Returns:
        SITKImage: Cropped SimpleITK image patch of size (patch_size, patch_size, patch_size),
        clipped to remain inside the input volume.
    """
    if center_voxel is None:
        raise ValueError("Center world coordinate is outside the image volume.")

    img_size = sitk_img.GetSize()  # (x, y, z)

    start = []
    size = []
    half = patch_size // 2

    for i in range(3):
        dim = img_size[i]
        # If the image dimension is smaller than patch_size, just take the whole dimension.
        if dim <= patch_size:
            start_i = 0
            size_i = dim
        else:
            start_i = max(0, center_voxel[i] - half)
            max_start = dim - patch_size
            if start_i > max_start:
                start_i = max_start
            size_i = patch_size
        start.append(int(start_i))
        size.append(int(size_i))

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start)
    roi_filter.SetSize(size)
    patch = roi_filter.Execute(sitk_img)
    return patch


def patch_sitk_to_numpy(patch_sitk: SITKImage) -> np.ndarray:
    """
    Convert a SimpleITK image patch to a NumPy array.

    Args:
        patch_sitk (SITKImage): 3D patch as a SimpleITK image.

    Returns:
        np.ndarray: Patch data as a float32 NumPy array with shape (z, y, x).
    """
    return sitk.GetArrayFromImage(patch_sitk).astype(np.float32)


def coords_to_m1p1(vox_coords, vol_size) -> tuple:
    """
    Normalize voxel coordinates to the range [-1, 1] along each axis.

    Args:
        vox_coords (tuple): Voxel coordinates (x, y, z) in index space.
        vol_size (tuple): Volume size (size_x, size_y, size_z).

    Returns:
        tuple: Normalized coordinates (x, y, z) as floats in the range [-1, 1].
    """
    x, y, z = vox_coords
    sx, sy, sz = vol_size
    return (
        2.0 * (x / (sx - 1)) - 1.0,
        2.0 * (y / (sy - 1)) - 1.0,
        2.0 * (z / (sz - 1)) - 1.0,
    )


if __name__ == "__main__":

    def _test_dicom_pixel_to_world():
        # Create a fake DICOM dataset with simple, axis-aligned geometry
        ds = pydicom.Dataset()
        ds.ImagePositionPatient = [10.0, 20.0, 30.0]  # origin
        # row_dir = (1, 0, 0), col_dir = (0, 1, 0)
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        # PixelSpacing = [row_spacing, col_spacing]
        ds.PixelSpacing = [2.0, 3.0]  # row_spacing=2 mm, col_spacing=3 mm

        # (x=0, y=0) should map exactly to origin
        p00 = dicom_pixel_to_world(ds, x=0, y=0)
        assert np.allclose(p00, [10.0, 20.0, 30.0]), f"p00={p00}"

        # (x=1, y=0): +1 column → +3 mm along col_dir = (0,1,0)
        p10 = dicom_pixel_to_world(ds, x=1, y=0)
        assert np.allclose(p10, [10.0, 23.0, 30.0]), f"p10={p10}"

        # (x=0, y=1): +1 row → +2 mm along row_dir = (1,0,0)
        p01 = dicom_pixel_to_world(ds, x=0, y=1)
        assert np.allclose(p01, [12.0, 20.0, 30.0]), f"p01={p01}"

        print("✅ _test_dicom_pixel_to_world passed")

    def _test_resample_image():
        # Create a synthetic 3D image: size (x, y, z) = (10, 20, 30)
        img = sitk.Image(10, 20, 30, sitk.sitkFloat32)
        img.SetSpacing((2.0, 2.0, 2.0))  # spacing (x, y, z)

        new_spacing = (1.0, 1.0, 1.0)
        img_resampled = resample_image(img, new_spacing=new_spacing, is_label=False)

        # Check that spacing was updated
        assert np.allclose(
            img_resampled.GetSpacing(), new_spacing
        ), f"spacing={img_resampled.GetSpacing()}"

        # Expected new size: original_size * (old_spacing / new_spacing)
        expected_size = (
            int(round(10 * (2.0 / 1.0))),
            int(round(20 * (2.0 / 1.0))),
            int(round(30 * (2.0 / 1.0))),
        )
        assert (
            img_resampled.GetSize() == expected_size
        ), f"size={img_resampled.GetSize()}, expected={expected_size}"

        print("✅ _test_resample_image passed")

    def _test_normalize_intensity():
        # ---- CT branch ----
        # values should be clipped to [-100, 700] and scaled to [0, 1]
        ct_values = np.array([-200.0, -100.0, 300.0, 700.0, 800.0], dtype=np.float32)
        ct_array = ct_values.reshape(1, 1, -1)
        ct_img = sitk.GetImageFromArray(ct_array)

        ct_img_norm = normalize_intensity(ct_img, modality="CT")
        ct_norm_array = sitk.GetArrayFromImage(ct_img_norm)
        ct_flat = ct_norm_array.ravel()

        # Shape is preserved
        assert ct_norm_array.shape == ct_array.shape, "CT shape mismatch"

        # Values should be in [0, 1] after clipping and scaling
        assert ct_flat.min() >= -1e-6, f"CT min out of range: {ct_flat.min()}"
        assert ct_flat.max() <= 1.0 + 1e-6, f"CT max out of range: {ct_flat.max()}"

        # ---- MRI branch ----
        # percentile clipping then min-max scaling → values in [0, 1]
        mri_values = np.linspace(0.0, 100.0, num=50, dtype=np.float32)
        mri_array = mri_values.reshape(1, 5, 10)
        mri_img = sitk.GetImageFromArray(mri_array)
        mri_img_norm = normalize_intensity(mri_img, modality="MRI")
        mri_norm_array = sitk.GetArrayFromImage(mri_img_norm)
        mri_flat = mri_norm_array.ravel()

        # Shape is preserved
        assert mri_norm_array.shape == mri_array.shape, "MRI shape mismatch"

        # Values should also be in [0, 1] after percentile clipping + scaling
        assert mri_flat.min() >= -1e-6, f"MRI min out of range: {mri_flat.min()}"
        assert mri_flat.max() <= 1.0 + 1e-6, f"MRI max out of range: {mri_flat.max()}"

        print("✅ _test_normalize_intensity passed")

    def _test_jitter_coords():
        # Deterministic jitter for the test
        np.random.seed(0)

        vox_coord = np.array([10.0, 20.0, 30.0], dtype=float)
        max_jitter = 5.0
        jittered = jitter_coords(vox_coord, max_jitter=max_jitter)

        assert isinstance(jittered, tuple), f"type={type(jittered)}"
        assert len(jittered) == 3, f"len={len(jittered)}"

        jittered_arr = np.asarray(jittered, dtype=float)
        diffs = jittered_arr - vox_coord

        # Each component should be within [-max_jitter, max_jitter]
        assert np.all(diffs >= -max_jitter - 1e-6), f"diffs lower bound: {diffs}"
        assert np.all(diffs <= max_jitter + 1e-6), f"diffs upper bound: {diffs}"

        # With a fixed seed, we also know it shouldn't be exactly the same coord
        assert not np.allclose(jittered_arr, vox_coord), "jitter had no effect"

        print("✅ _test_jitter_coords passed")

    def _test_world_to_voxel():
        # Simple 3D image: size (x, y, z) = (10, 20, 30), spacing (1,1,1), origin (0,0,0)
        img = sitk.Image(10, 20, 30, sitk.sitkFloat32)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((0.0, 0.0, 0.0))

        inside_world = np.array([2.1, 5.9, 10.4], dtype=float)
        idx = world_to_voxel(img, inside_world)
        # Continuous index ~ (2.1, 5.9, 10.4) → rounded (2, 6, 10)
        assert idx == (2, 6, 10), f"idx={idx}"

        # Clearly outside: negative coordinate
        outside_world_1 = np.array([-1.0, 0.0, 0.0], dtype=float)
        assert world_to_voxel(img, outside_world_1) is None

        # Clearly outside: beyond max x
        outside_world_2 = np.array([100.0, 0.0, 0.0], dtype=float)
        assert world_to_voxel(img, outside_world_2) is None

        print("✅ _test_world_to_voxel passed")

    def _test_extract_patch_sitk():
        # Case 1: volume larger than patch_size in all dims
        img = sitk.Image(20, 30, 40, sitk.sitkFloat32)  # (x=20, y=30, z=40)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((0.0, 0.0, 0.0))

        center_voxel = np.array([10.0, 15.0, 20.0], dtype=float)
        patch_size = 8
        patch = extract_patch_sitk(
            img, center_voxel=center_voxel, patch_size=patch_size
        )

        # In index space, patch size should be exactly (8,8,8)
        assert patch.GetSize() == (
            patch_size,
            patch_size,
            patch_size,
        ), f"patch size={patch.GetSize()}"

        # Case 2: volume smaller than patch_size in some dims
        img_small = sitk.Image(10, 8, 6, sitk.sitkFloat32)  # (x=10, y=8, z=6)
        img_small.SetSpacing((1.0, 1.0, 1.0))
        img_small.SetOrigin((0.0, 0.0, 0.0))

        center_voxel_small = np.array([5.0, 4.0, 3.0], dtype=float)  # roughly center
        patch_large = extract_patch_sitk(
            img_small, center_voxel=center_voxel_small, patch_size=16
        )

        # Patch should be clipped to image size: can't exceed original dims
        assert (
            patch_large.GetSize() == img_small.GetSize()
        ), f"patch_large size={patch_large.GetSize()}, img_small size={img_small.GetSize()}"

        print("✅ _test_extract_patch_sitk passed")

    def _test_patch_sitk_to_numpy():
        # Simple 3D image with known size
        img = sitk.Image(5, 6, 7, sitk.sitkFloat32)  # (x=5, y=6, z=7)
        arr = patch_sitk_to_numpy(img)

        # SimpleITK arrays are (z, y, x) => (7, 6, 5)
        assert arr.shape == (7, 6, 5), f"shape={arr.shape}"
        assert arr.dtype == np.float32, f"dtype={arr.dtype}"

        print("✅ _test_patch_sitk_to_numpy passed")

    def _test_coords_to_m1p1():
        vol_size = (10, 20, 30)  # (x, y, z)

        # ---- minimum corner ----
        c_min = (0, 0, 0)
        out_min = coords_to_m1p1(c_min, vol_size)
        assert out_min == (-1.0, -1.0, -1.0), f"min={out_min}"

        # ---- maximum corner ----
        c_max = (vol_size[0] - 1, vol_size[1] - 1, vol_size[2] - 1)
        out_max = coords_to_m1p1(c_max, vol_size)
        assert out_max == (1.0, 1.0, 1.0), f"max={out_max}"

        # ---- center of volume ----
        c_center = (
            (vol_size[0] - 1) / 2,
            (vol_size[1] - 1) / 2,
            (vol_size[2] - 1) / 2,
        )
        out_center = coords_to_m1p1(c_center, vol_size)
        assert np.allclose(out_center, (0.0, 0.0, 0.0)), f"center={out_center}"

        # ---- output type ----
        assert isinstance(out_center, tuple), f"type={type(out_center)}"
        assert len(out_center) == 3, f"len={len(out_center)}"

        print("✅ _test_coords_to_m1p1 passed")

    # Run tests
    _test_dicom_pixel_to_world()
    _test_resample_image()
    _test_normalize_intensity()

    _test_jitter_coords()
    _test_world_to_voxel()
    _test_extract_patch_sitk()
    _test_patch_sitk_to_numpy()
    _test_coords_to_m1p1()

    print("🎉 All tests passed.")
