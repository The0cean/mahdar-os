"""Mahdar's app icon in Mizan's style: black square, orange (#FF4500) "م" with an underscore.
Writes a 1024 px PNG; then run `npx tauri icon <png>` to make every size."""
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/tahomabd.ttf"
ORANGE = (255, 69, 0, 255)


def icon(size):
    img = Image.new("RGBA", (size, size), (10, 10, 10, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, int(size * 0.70))
    box = d.textbbox((0, 0), "م", font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    x = (size - w) / 2 - box[0]
    y = size * 0.42 - h / 2 - box[1]
    d.text((x, y), "م", font=f, fill=ORANGE)
    bar = max(1, size // 16)
    y0 = int(size * 0.80)
    d.rectangle([int(size * 0.22), y0, int(size * 0.70), y0 + bar], fill=ORANGE)
    return img


icon(1024).save(sys.argv[1])
print("wrote", sys.argv[1])
