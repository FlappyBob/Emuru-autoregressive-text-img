import argparse
import torch
from tqdm import tqdm
from loguru import logger
from accelerate import Accelerator

from models.htr import HTR
from models.smooth_ce import SmoothCrossEntropyLoss
from custom_datasets import ours_DataLoaderManager
import evaluate


@torch.no_grad()
def run_eval(eval_loader, alphabet, htr, accelerator, weight_dtype):
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()

    cer_metric = evaluate.load("cer")
    ce_loss = SmoothCrossEntropyLoss(tgt_pad_idx=0)

    total_loss = 0
    total_steps = 0

    for batch in tqdm(eval_loader, desc="Evaluating"):
        images = batch["bw"].to(weight_dtype)

        text_logits_s2s = batch["text_logits_s2s"]
        tgt_mask = batch["tgt_key_mask"]
        tgt_key_padding_mask = batch["tgt_key_padding_mask"]

        with accelerator.autocast():
            output = htr_model(images, text_logits_s2s[:, :-1], tgt_mask, tgt_key_padding_mask[:, :-1])
            loss = ce_loss(output, text_logits_s2s[:, 1:])
            total_loss += accelerator.gather(loss).mean().item()
            total_steps += 1

        # decode CER
        predicted_logits = torch.argmax(output, dim=2)
        predicted_texts = alphabet.decode(predicted_logits, [alphabet.eos])
        gt_texts = alphabet.decode(text_logits_s2s[:, 1:], [alphabet.eos])
        cer_metric.add_batch(predictions=predicted_texts, references=gt_texts)

    final_cer = cer_metric.compute()
    final_loss = total_loss / total_steps

    accelerator.print(f"\n🔥 FINAL EVAL — Loss={final_loss:.4f} | CER={final_cer:.4f}")
    return final_loss, final_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing config.json + model.safetensors")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # load model
    logger.info(f"Loading pretrained HTR model from: {args.model_dir}")
    htr = HTR.from_pretrained(args.model_dir)

    # dataset
    data_loader = ours_DataLoaderManager(
        train_pattern=None,
        eval_pattern=("https://hf-mirror.com/datasets/blowing-up-groundhogs/font-square-pretrain-20M/resolve/main/{000000..000010}.tar"),
        train_batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    eval_loader = data_loader.create_dataset("eval", "htr")
    htr, eval_loader = accelerator.prepare(htr, eval_loader)

    # precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    run_eval(eval_loader, data_loader.alphabet, htr, accelerator, weight_dtype)


if __name__ == "__main__":
    main()
