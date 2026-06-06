import numpy as np
import pytest
import tensorflow as tf

from data.oxford_pets import OxfordPetsSegmentationDataset


#ASK: MANY
def _write_test_data(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations_dir = tmp_path / "annotations" / "trimaps"
    annotations_dir.mkdir(parents=True)

    image = tf.zeros((8, 10, 3), dtype=tf.uint8)
    encoded_image = tf.io.encode_jpeg(image)
    tf.io.write_file(str(images_dir / "Abyssinian_1.jpg"), encoded_image)
    tf.io.write_file(str(images_dir / "Abyssinian_2.jpg"), encoded_image)
    tf.io.write_file(str(images_dir / "image_without_mask.jpg"), encoded_image)

    mask = tf.constant(
        [
            [[1], [2], [3], [1]],
            [[2], [3], [1], [2]]
        ],
        dtype=tf.uint8
    )
    encoded_mask = tf.io.encode_png(mask)
    tf.io.write_file(str(annotations_dir / "Abyssinian_1.png"), encoded_mask)
    tf.io.write_file(str(annotations_dir / "Abyssinian_2.png"), encoded_mask)
    tf.io.write_file(str(annotations_dir / "mask_without_image.png"), encoded_mask)
    return images_dir, annotations_dir


def test_sample_ds_returns_only_matched_image_and_mask_pairs(tmp_path):
    """验证样本列表只包含图片和掩码都存在的配对。"""
    images_dir, annotations_dir = _write_test_data(tmp_path)
    dataset = OxfordPetsSegmentationDataset(
        images_path=images_dir,
        annotations_path=annotations_dir
    )

    samples = list(dataset.sample_ds().as_numpy_iterator())

    assert len(samples) == 2
    assert samples[0]["image_path"].decode("utf-8").endswith("Abyssinian_1.jpg")
    assert samples[0]["mask_path"].decode("utf-8").endswith("Abyssinian_1.png")


def test_training_ds_builds_segmentation_images_and_masks(tmp_path):
    """验证训练数据集会输出调整尺寸后的图片和类别掩码。"""
    images_dir, annotations_dir = _write_test_data(tmp_path)
    dataset = OxfordPetsSegmentationDataset(
        images_path=images_dir,
        annotations_path=annotations_dir,
        image_size=(6, 6)
    )

    images, masks = next(iter(dataset.training_ds(batch_size=2)))

    assert images.shape == (2, 6, 6, 3)
    assert masks.shape == (2, 6, 6, 1)
    assert images.dtype == tf.float32
    assert masks.dtype == tf.int32
    np.testing.assert_array_equal(np.unique(masks.numpy()), [0, 1, 2])


def test_training_ds_raises_error_when_no_matched_samples(tmp_path):
    """验证没有匹配样本时会直接报错，避免静默训练空数据。"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations_dir = tmp_path / "annotations" / "trimaps"
    annotations_dir.mkdir(parents=True)
    dataset = OxfordPetsSegmentationDataset(
        images_path=images_dir,
        annotations_path=annotations_dir
    )

    with pytest.raises(ValueError, match="没有找到匹配样本"):
        dataset.training_ds(batch_size=2)
