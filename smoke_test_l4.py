"""L4 应用层模型全量验证脚本。

测试所有 8 个模型：ViT/ResNet/YOLO/BERT/U-Net/GPT/DCGAN/Multimodal。
用法：python smoke_test_l4_models.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import zerotorch as zt

PASS = 0
FAIL = 0

def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        print(f'  [PASS] {name} {detail}')
        PASS += 1
    else:
        print(f'  [FAIL] {name} {detail}')
        FAIL += 1


def test_vit():
    print('\n== ViT-lite 图像分类 ==')
    from zerotorch.models import ViTLite
    model = ViTLite(img_size=32, patch_size=4, in_channels=1, num_classes=3,
                    d_model=64, num_heads=4, num_layers=2)
    x = zt.tensor(np.random.randn(2, 1, 32, 32).astype(np.float64))
    out = model(x)
    check('前向形状', out.data.shape == (2, 3), str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_resnet():
    print('\n== ResNet-lite 图像分类 ==')
    from zerotorch.models import ResNetLite
    model = ResNetLite(in_channels=1, num_classes=3, channels=[16, 32, 64, 128], img_size=32)
    x = zt.tensor(np.random.randn(2, 1, 32, 32).astype(np.float64))
    out = model(x)
    check('前向形状', out.data.shape == (2, 3), str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_yolo():
    print('\n== YOLOv3-lite 目标检测 ==')
    from zerotorch.models import YOLOv3Lite
    model = YOLOv3Lite(in_channels=1, num_classes=3, anchors=[[10,10],[20,20],[40,40]], grid_size=8)
    x = zt.tensor(np.random.randn(2, 1, 32, 32).astype(np.float64))
    out = model(x)
    check('前向形状', out.data.shape[0] == 2, str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_bert():
    print('\n== BERT-lite 文本理解 ==')
    from zerotorch.models import BERTLite
    model = BERTLite(vocab_size=500, d_model=64, num_heads=4, num_layers=2, max_seq_len=16)
    input_ids = zt.tensor(np.random.randint(0, 500, size=(2, 16)).astype(np.int64))
    mlm_logits, seq_out = model(input_ids)
    check('前向形状', mlm_logits.data.shape == (2, 16, 500), str(mlm_logits.data.shape))
    mlm_logits.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_unet():
    print('\n== U-Net 图像分割 ==')
    from zerotorch.models import UNetLite
    model = UNetLite(in_channels=1, num_classes=3, base_channels=8)
    x = zt.tensor(np.random.randn(2, 1, 32, 32).astype(np.float64))
    out = model(x)
    check('前向形状', out.data.shape == (2, 3, 32, 32), str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_gpt():
    print('\n== GPT-lite 文本生成 ==')
    from zerotorch.models import GPTLite
    model = GPTLite(vocab_size=500, d_model=64, num_heads=4, num_layers=2, max_seq_len=32)
    input_ids = zt.tensor(np.random.randint(0, 500, size=(2, 16)).astype(np.int64))
    out = model(input_ids)
    check('前向形状', out.data.shape == (2, 16, 500), str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    # 自回归生成
    generated = model.generate([5, 6], max_new_tokens=5, temperature=1.0)
    check('自回归生成长度', len(generated) == 7, f'生成{len(generated)}个token')
    print('  参数量:', model.num_parameters())


def test_dcgan():
    print('\n== DCGAN-lite 图像生成 ==')
    from zerotorch.models import DCGANLite
    model = DCGANLite(z_dim=64)
    # 生成器
    z = zt.tensor(np.random.randn(2, 64).astype(np.float64))
    fake = model.generator(z)
    check('生成器输出形状', fake.data.shape == (2, 1, 32, 32), str(fake.data.shape))
    check('生成器输出范围[0,1]', fake.data.min() >= 0 and fake.data.max() <= 1,
          f'[{fake.data.min():.3f}, {fake.data.max():.3f}]')
    # 判别器
    d_out = model.discriminator(fake)
    check('判别器输出形状', d_out.data.shape == (2,), str(d_out.data.shape))
    check('判别器输出范围[0,1]', d_out.data.min() >= 0 and d_out.data.max() <= 1,
          f'[{d_out.data.min():.3f}, {d_out.data.max():.3f}]')
    # 生成推理
    fake_imgs = model.generate(batch_size=4)
    check('generate 输出形状', fake_imgs.data.shape == (4, 1, 32, 32), str(fake_imgs.data.shape))
    print('  生成器参数量:', model.generator.num_parameters())
    print('  判别器参数量:', model.discriminator.num_parameters())


def test_multimodal():
    print('\n== Multimodal-lite 图文匹配 ==')
    from zerotorch.models import MultimodalLite
    model = MultimodalLite(vocab_size=500, img_dim=64, txt_dim=32, num_classes=2)
    images = zt.tensor(np.random.randn(2, 1, 32, 32).astype(np.float64))
    input_ids = zt.tensor(np.random.randint(0, 500, size=(2, 10)).astype(np.int64))
    out = model(images, input_ids)
    check('前向形状', out.data.shape == (2, 2), str(out.data.shape))
    out.sum().backward()
    check('梯度可反传', all(p.grad is not None for p in model.parameters()))
    print('  参数量:', model.num_parameters())


def test_registry():
    print('\n== MODEL_REGISTRY 注册机制 ==')
    from zerotorch.models import MODEL_REGISTRY
    expected = ['vit_lite', 'resnet_lite', 'yolov3_lite', 'bert_lite',
                'unet_lite', 'gpt_lite', 'dcgan_lite', 'multimodal_lite']
    for name in expected:
        check(f'注册 {name}', name in MODEL_REGISTRY)
    print('  已注册模型:', list(MODEL_REGISTRY.keys()))


def main():
    print('=' * 60)
    print('ZeroTorch L4 应用层模型全量验证')
    print('=' * 60)

    test_vit()
    test_resnet()
    test_yolo()
    test_bert()
    test_unet()
    test_gpt()
    test_dcgan()
    test_multimodal()
    test_registry()

    print('\n' + '=' * 60)
    print(f'结果: {PASS} PASS, {FAIL} FAIL')
    print('=' * 60)
    return FAIL


if __name__ == '__main__':
    sys.exit(main())
