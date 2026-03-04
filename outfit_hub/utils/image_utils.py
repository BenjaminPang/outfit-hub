from PIL import Image
import io
import os


def process_and_pad_image(img_path: str, target_size=291, bg_color=(255, 255, 255)):
    """
    等比例缩放图片并在短边填充，确保不形变，并将透明背景转为指定背景色。
    """
    if not os.path.exists(img_path):
        return None
    
    try:
        with Image.open(img_path) as img:
            # 1. 处理 PNG 透明通道并转为 RGB
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, bg_color + (255,))
                img = Image.alpha_composite(background, img).convert("RGB")
            else:
                img = img.convert("RGB")

            # 2. 计算等比例缩放尺寸
            w, h = img.size
            if (w, h) != (target_size, target_size):
                ratio = target_size / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

                # 3. 填充到 291x291 白色画布
                new_img = Image.new("RGB", (target_size, target_size), bg_color)
                paste_x = (target_size - new_w) // 2
                paste_y = (target_size - new_h) // 2
                new_img.paste(img, (paste_x, paste_y))
            else:
                new_img = img
            
            # 4. 转换为 JPEG 字节流
            img_byte_arr = io.BytesIO()
            new_img.save(img_byte_arr, format='JPEG', quality=95)
            return img_byte_arr.getvalue()
    except Exception as e:
        print(f"Process image {img_path} failed: {e}")
        return None