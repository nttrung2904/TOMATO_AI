from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
fb_dir = ROOT / 'static' / 'feedback' / 'confirmed_tomato'
samples_dir = ROOT / 'static' / 'images' / 'tomato_samples'

if not fb_dir.exists():
    print(f"Không tìm thấy thư mục feedback: {fb_dir}")
    raise SystemExit(1)

samples_dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {'.png', '.jpg', '.jpeg'}
moved = []

for f in sorted(fb_dir.iterdir()):
    if f.suffix.lower() not in ALLOWED_EXTS:
        continue
    
    target = samples_dir / f.name
    # Xử lý trùng tên file
    if target.exists():
        i = 1
        while (samples_dir / f"{f.stem}_{i}{f.suffix}").exists():
            i += 1
        target = samples_dir / f"{f.stem}_{i}{f.suffix}"
    
    shutil.copy2(f, target)
    moved.append((f.name, target.name))

if moved:
    print(f"Đã di chuyển {len(moved)} ảnh:")
    for src, dst in moved:
        print(f"  {src} -> {dst}")
else:
    print("Không tìm thấy ảnh nào để di chuyển.")

print('Hoàn thành.')
