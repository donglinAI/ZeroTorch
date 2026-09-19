"""ZeroTorch 统一推理接口。

用法：
    python examples/inference.py --model resnet --weights resnet_weights.pt
    python examples/inference.py --model vit --weights vit_weights.pt
    python examples/inference.py --model yolo --weights yolo_weights.pt
"""
import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import zerotorch as zt


# 模型注册表：模型名 → (模型类, 配置, 输入说明)
MODEL_REGISTRY = {
    'resnet': {
        'class': zt.models.ResNetLite,
        'config': dict(in_channels=1, num_classes=10, channels=[8,16,32,64], img_size=32),
        'input_type': 'image',
        'input_shape': (1, 32, 32),
        'task': 'classification',
        'classes': [str(i) for i in range(10)],
    },
    'vit': {
        'class': zt.models.ViTLite,
        'config': dict(img_size=32, patch_size=4, in_channels=1, num_classes=10, d_model=16, num_heads=2, num_layers=1),
        'input_type': 'image',
        'input_shape': (1, 32, 32),
        'task': 'classification',
        'classes': [str(i) for i in range(10)],
    },
    'yolo': {
        'class': zt.models.YOLOv3Lite,
        'config': dict(in_channels=1, num_classes=2, anchors=[[10,10],[20,20],[40,40]], grid_size=8),
        'input_type': 'image',
        'input_shape': (1, 32, 32),
        'task': 'detection',
        'classes': ['circle', 'cross'],
    },
}


def load_model(model_name, weights_path=None):
    """加载模型 + 权重。"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}, 可选: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[model_name]
    model = cfg['class'](**cfg['config'])

    if weights_path:
        zt.serialization.load(model, weights_path)
        print(f"已加载权重: {weights_path}")

    return model, cfg


def predict_image(model, cfg, input_tensor):
    """图像类模型推理（分类/检测）。"""
    model.eval()
    output = model(input_tensor)

    task = cfg.get('task', 'classification')

    if task == 'classification':
        pred_cls = output.data.argmax(axis=1)
        print(f"\n推理结果:")
        for i, cls_idx in enumerate(pred_cls):
            cls_name = cfg['classes'][cls_idx] if 'classes' in cfg else f"class_{cls_idx}"
            prob = float(output.data[i, cls_idx])
            print(f"  样本 {i}: {cls_name} (置信度 {prob:.4f})")

    elif task == 'detection':
        print(f"\n检测输出: {output.data.shape}")
        print("  推理流程: forward() → decode_predictions() → NMS()")

    return output


def main():
    parser = argparse.ArgumentParser(description='ZeroTorch 统一推理接口')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()),
                        help='模型名称')
    parser.add_argument('--weights', type=str, default=None,
                        help='权重文件路径 (.pt)')
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"ZeroTorch 推理引擎")
    print(f"模型: {args.model}")
    print(f"=" * 60)

    # 加载模型
    model, cfg = load_model(args.model, args.weights)
    print(f"参数量: {model.num_parameters():,}")

    # 推理（用随机输入演示）
    x = np.random.randn(2, *cfg['input_shape']).astype(np.float64)
    input_tensor = zt.tensor(x)
    predict_image(model, cfg, input_tensor)


if __name__ == '__main__':
    main()
