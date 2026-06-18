"""测试 iRevNet 模型加载和推理"""
import sys
import os
sys.path.insert(0, '/home/wyt/AsymCDC/clipper-asymcc/run')

from clipper_deploy import load_irevnet_model, predict
import io
from PIL import Image

def test_model():
    print("=" * 50)
    print("测试 1: 模型加载")
    print("=" * 50)
    
    model_path = '/home/wyt/AsymCDC/model/src/train/checkpoint/cifar10/i-revnet-55.t7'
    
    try:
        model = load_irevnet_model(model_path)
        print("✓ 模型加载成功")
        print(f"  模型类型: {type(model)}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 50)
    print("测试 2: 模型推理")
    print("=" * 50)
    
    # 创建测试图像
    img = Image.new('RGB', (32, 32), color=(100, 150, 200))
    img_byte = io.BytesIO()
    img.save(img_byte, format='JPEG')
    img_bytes = img_byte.getvalue()
    
    print(f"  测试图像大小: {len(img_bytes)} bytes")
    
    try:
        result = predict(model, [img_bytes])
        print("✓ 推理成功!")
        print(f"  结果类型: {type(result)}")
        print(f"  结果长度: {len(result)}")
        if isinstance(result[0], list):
            # 反序列化看实际内容
            import pickle
            out_tensor = pickle.loads(result[0][0])
            out_bij = pickle.loads(result[0][1])
            print(f"  输出形状: {out_tensor.shape}")
            print(f"  输出bij形状: {out_bij.shape}")
            print(f"  输出预测类别: {out_tensor.argmax(dim=1).item()}")
        else:
            print(f"  结果内容: {result}")
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
