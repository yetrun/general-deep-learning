import tensorflow as tf

from data.image_classification import ImageClassificationDirectoryDataset


def _write_classification_data(tmp_path):
    image = tf.zeros((8, 10, 3), dtype=tf.uint8)
    encoded_image = tf.io.encode_jpeg(image)
    for split in ["train", "val", "test"]:
        for class_name in ["cat", "dog"]:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            tf.io.write_file(str(class_dir / f"{class_name}_1.jpg"), encoded_image)
            tf.io.write_file(str(class_dir / f"{class_name}_2.jpg"), encoded_image)
    return tmp_path / "train", tmp_path / "val", tmp_path / "test"


def test_training_ds_builds_binary_classification_images_and_labels(tmp_path):
    """验证目录分类训练数据集会输出调整尺寸后的图片和二分类标签。"""
    train_dir, validation_dir, test_dir = _write_classification_data(tmp_path)
    dataset = ImageClassificationDirectoryDataset(
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(6, 6)
    )

    images, labels = next(iter(dataset.training_ds(batch_size=2)))

    assert images.shape == (2, 6, 6, 3)
    assert labels.shape == (2, 1)
    assert images.dtype == tf.float32
    assert labels.dtype == tf.float32


def test_class_names_returns_directory_class_names(tmp_path):
    """验证目录分类数据集会按目录名返回类别名称。"""
    train_dir, validation_dir, test_dir = _write_classification_data(tmp_path)
    dataset = ImageClassificationDirectoryDataset(
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(6, 6)
    )

    assert dataset.class_names() == ["cat", "dog"]


def test_validation_and_test_ds_read_their_own_directories(tmp_path):
    """验证验证集和测试集会分别从对应目录读取图片。"""
    train_dir, validation_dir, test_dir = _write_classification_data(tmp_path)
    dataset = ImageClassificationDirectoryDataset(
        train_path=train_dir,
        validation_path=validation_dir,
        test_path=test_dir,
        image_size=(6, 6)
    )

    validation_images, validation_labels = next(iter(dataset.validation_ds(batch_size=2)))
    test_images, test_labels = next(iter(dataset.test_ds(batch_size=2)))

    assert validation_images.shape == (2, 6, 6, 3)
    assert validation_labels.shape == (2, 1)
    assert test_images.shape == (2, 6, 6, 3)
    assert test_labels.shape == (2, 1)
