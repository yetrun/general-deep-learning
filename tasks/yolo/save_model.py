from tasks.yolo.train import resolve_pipeline


def main():
    pip = resolve_pipeline()
    model_path = pip.save_inference_model()
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
