# main.py
import torch
from scr.pipeline.voucher_pipeline import VoucherPipeline
from scr.utils.config_loader import load_config

if __name__ == '__main__':
    config = load_config('config/settings.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dirs = {
        'vouchers_a_segmentar': config['paths']['vouchers_a_segmentar'],
        'single_voucher': config['paths']['single_voucher'],
        'no_voucher': config['paths']['output_no_voucher_dir'],
        'validated_voucher': config['paths']['validated_voucher_dir'],
        'outputs': config['paths']['outputs_json_dir'],
    }
    seg_config = {
        'model_name': config['segmentation']['model_name'],
        'checkpoint': config['segmentation']['checkpoint'],
    }
    val_config = {
        'checkpoint': config['validation']['checkpoint'],
        'labels': config['validation']['labels'],
        'confidence_threshold': config['validation']['confidence_threshold']
    }
    pipeline = VoucherPipeline(seg_config, val_config, config['ocr']['method'], base_dirs, device)
    pipeline.run()
