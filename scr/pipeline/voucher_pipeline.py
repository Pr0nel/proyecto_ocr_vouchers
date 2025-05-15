# scr/pipeline/voucher_pipeline.py
import json
import torch
import shutil
from PIL import Image
from pathlib import Path
from scr.segmentation.voucher_segmentation import Segmenter
from scr.validation.voucher_validation import Validator
from scr.ocr.voucher_ocr import OCRExtractor

class VoucherPipeline:
    def __init__(self,
                 seg_config: dict,
                 val_config: dict,
                 ocr_method: str,
                 base_dirs: dict,
                 device: torch.device):
        self.segmenter = Segmenter(**seg_config, device=device)
        self.validator = Validator(**val_config, device=device)
        self.ocr = OCRExtractor(method=ocr_method)
        self.dirs = {k: Path(v) for k, v in base_dirs.items()}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def run(self):
        for img_file in self.dirs['vouchers_a_segmentar'].iterdir():
            segments = self.segmenter.segment(img_file)
            for idx, seg in enumerate(segments):
                seg_path = self.dirs['single_voucher'] / f"{img_file.stem}_voucher_{idx}.png"
                seg.save(seg_path)

        for img_file in list(self.dirs['single_voucher'].iterdir()):
            img = Image.open(img_file)
            if self.validator.is_voucher(img):
                dest = self.dirs['validated_voucher'] / f"{img_file.stem}_v{img_file.suffix}"
                shutil.move(str(img_file), dest)
                text = self.ocr.extract(Image.open(dest))
                json_path = self.dirs['outputs'] / f"{dest.stem}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({'text': text}, f, ensure_ascii=False)
            else:
                dest = self.dirs['no_voucher'] / img_file.name
                shutil.move(str(img_file), dest)
