from PIL import Image
import os

def main():

    input_dir =  "/media/zyserver/data16t/lpd/ddrm/images"
    output_dir = "/media/zyserver/data16t/lpd/ddrm/images_noisy"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}:
            in_path = os.path.join(input_dir, fname)
            try:
                with Image.open(in_path) as img:
                    img = img.convert("RGB")
                    img_resized = img.resize((512, 512))
                    out_path = os.path.join(output_dir, fname)
                    img_resized.save(out_path)
                    print("Saved:", out_path)
            except Exception as e:
                print("Failed:", in_path, e)

if __name__ == "__main__":
    main()