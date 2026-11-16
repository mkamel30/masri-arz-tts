#!/usr/bin/env python3
"""
End-to-End Automated Pipeline for Masri Egyptian Arabic (arz) 2-Speaker Piper VITS TTS
Repo: https://github.com/mkamel30/masri-arz-tts.git

Stages:
  1. corpus   -> Builds 16,000 normalized Egyptian Arabic sentences (Ahmed + Mariam)
  2. generate -> Distills 32,000 WAV clips via VoiceTut-TTS onto Google Drive (auto-resumes)
  3. train    -> Installs OHF-Voice/piper1-gpl, trains 2-speaker VITS, exports .onnx + .onnx.json
  4. all      -> Runs stages 1 -> 2 -> 3 sequentially
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ==============================================================================
# 1. EGYPTIAN ARABIC TEXT NORMALIZER (Digits, Clock Times, Percentages)
# ==============================================================================
ONES = ["صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة"]
TEENS = ["عشرة", "حداشر", "اتناشر", "تلتاشر", "أربعتاشر", "خمستاشر", "ستاشر", "سبعتاشر", "تمنتاشر", "تسعتاشر"]
TENS = ["", "", "عشرين", "تلاتين", "أربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين"]
HUNDREDS = ["", "مية", "ميتين", "تلتمية", "ربعمية", "خمسمية", "ستمية", "سبعمية", "تمنمية", "تسعمية"]


def int_to_egyptian(n: int) -> str:
    if n < 0:
        return "سالب " + int_to_egyptian(-n)
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] if o == 0 else f"{ONES[o]} و{TENS[t]}"
    if n < 1000:
        h, rem = divmod(n, 100)
        return HUNDREDS[h] if rem == 0 else f"{HUNDREDS[h]} و{int_to_egyptian(rem)}"
    if n < 10000:
        th, rem = divmod(n, 1000)
        prefix = "ألف" if th == 1 else ("ألفين" if th == 2 else f"{ONES[th]} آلاف")
        return prefix if rem == 0 else f"{prefix} و{int_to_egyptian(rem)}"
    return str(n)


def normalize_egyptian_text(text: str) -> str:
    text = re.sub(r"\s*/\s*", "، ", text)

    def _time_repl(m):
        h, mins = int(m.group(1)), int(m.group(2))
        hw = int_to_egyptian(h)
        if mins == 0:
            return f"{hw} بالظبط"
        if mins == 15:
            return f"{hw} وربع"
        if mins == 30:
            return f"{hw} ونص"
        if mins == 45:
            return f"{int_to_egyptian(1 if h == 12 else h + 1)} إلا ربع"
        return f"{hw} و{int_to_egyptian(mins)} دقيقة"

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_repl, text)
    text = re.sub(r"\b(\d{1,3})\s*%", lambda m: f"{int_to_egyptian(int(m.group(1)))} في المية", text)
    text = re.sub(r"\b\d{1,4}\b", lambda m: int_to_egyptian(int(m.group(0))), text)
    return re.sub(r"\s+", " ", text).strip()


# ==============================================================================
# 2. 16,000-SENTENCE EGYPTIAN ARABIC CORPUS BUILDER
# ==============================================================================
GREETINGS = [
    "أهلاً وسهلاً يا باشا", "صباح الخير يا فندم", "مساء النور يا صاحبي",
    "إزيك عامل إيه النهاردة", "إزيك عاملة إيه يا أستاذة", "نورت الدنيا كلها",
    "أهلاً بيك في القاهرة", "عامل إيه في الشغل", "إيه الأخبار طمني عليك",
    "صباح الفل والقشطة", "مساء الفل يا معلم", "أهلاً يا ست الكل",
]

LOCATIONS = [
    "وسط البلد", "المعادي", "مدينة نصر", "مصر الجديدة", "الدقي", "المهندسين",
    "الزمالك", "التجمع الخامس", "الشيخ زايد", "أكتوبر", "إسكندرية", "السويس",
    "مطار القاهرة", "محطة المترو", "ميدان التحرير", "خان الخليلي", "الكورنيش",
    "البنك الأهلي", "الصيدلية", "السوبرماركت", "المستشفى", "الجامعة", "المكتب",
]

FOOD_DRINKS = [
    "فنجان قهوة مظبوط", "قهوة سادة من غير سكر", "شاي بالنعناع الأخضر",
    "كوباية شاي كشري", "طبق كشري كبير بالدقة", "ساندوتش طعمية سخنة",
    "طبق فول بالزيت الحار والطحينة", "ملوخية بالأرانب", "كباب وكفتة مشوية",
    "حواوشي سخن ومقرمش", "عصير مانجو فريش", "إزازة مية معدنية ساقعة",
    "رز بلبن بالمكسرات", "ساندوتش شاورما فراخ بالتومية", "كرواسون بالزبدة",
]

LOANWORDS = [
    "عندي ميتينج مهم أونلاين.", "ابعتلي اللوكيشن على الواتساب.",
    "شحن الموبايل فصل ومحتاج شاحن.", "الباسورد بتاع الواي فاي مش شغال.",
    "اللابتوب محتاج أبديت سريع.", "بعتلك الإيميل بالمرفقات.",
    "هدفع بالفيزا تاتش عند الكاشير.", "اطلب أوردر دليفري من الأبلكيشن.",
    "السيستم واقع النهاردة في الشركة.", "عملت شير للينك على الجروب.",
    "سجلت فويس ريكورد طويل.", "الطيارة فيها ديلاي نص ساعة.",
    "معايا كارت البوردنج والباسبور.", "الشقة فيها تكييف وواي فاي سريع.",
]

IDIOM_TAILS = [
    "عشان منتاخرش على الميعاد.", "لو سمحت من فضلك.", "بصراحة الموضوع يستاهل.",
    "ومفيش أي مشكلة خالص.", "إن شاء الله خير.", "ربنا يسهل الأمور.",
    "من غير أي تردد.", "على طول من غير لف ودوران.", "في أقرب وقت ممكن.",
    "عشان الدنيا زحمة موت برة.", "والحمد لله كله تمام.", "يا ريت ترد عليا بسرعة.",
]

TEMPLATES = [
    "لو سمحت يا أسطى، رايح {loc}، شغل العداد من فضلك.",
    "أروح إزاي {loc} من هنا، أمشي على طول ولا أخش يمين؟",
    "نزلني هنا على جنب قدام {loc} لو سمحت.",
    "المواصلات في {loc} زحمة موت، تعالى ناخد المترو أحسن.",
    "لف من اليوتيرن اللي جاي وهتلاقي {loc} على إيدك الشمال.",
    "أنا واقف قدام {loc} ومستنيك بقالي ربع ساعة.",
    "التاكسي خد مني {price} جنيه من {loc} لحد البيت.",
    "مساء الخير، هاتلي {food} لو سمحت.",
    "أنا جعان أوي، يلا نروح ناكل {food} في {loc}.",
    "الحساب كام يا كابتن؟ عايز أدفع كاش.",
    "بكام القميص الأبيض ده؟ ده غالي أوي يا باشا، اعملي خصم.",
    "آخرها كام من الآخر كده؟ أنا زبونك وبجيلك علطول.",
    "خلصانة على {price} جنيه، مبروك عليك يا فندم.",
    "الخدمة ممتازة والأكل طعمه تحفة، خلي الباقي عشانك.",
    "أنا بروح {loc} كل يوم الصبح الساعة {time}.",
    "إحنا بنشرب {food} سوا بعد ما نخلص الشغل.",
    "هي بتشتغل مهندسة برمجيات في شركة كبيرة في {loc}.",
    "بكرة إن شاء الله هروح {loc} بدري عشان أخلص الورق.",
    "هنشوفك إمتى يا صاحبي؟ هكلمك بالليل نرتب سوا.",
    "أنا ماشوفتوش إمبارح في {loc}، كان واخد إجازة.",
    "بصراحة ماعرفتش أوصل للعنوان عشان الموبايل فصل شحن.",
    "ماتقلقش وماتشغلش بالك، كبر دماغك والموضوع هيعدي.",
    "أنا مش عايز أخرج دلوقتي، الجو برد ومطر برة.",
    "هي مش عايزة تتأخر على الطيارة، لازم تمشي حالاً.",
    "عندي صداع شديد وزوري بيوجعني من إمبارح بالليل.",
    "لو سمحت فين أقرب صيدلية فاتحة أربعة وعشرين ساعة؟",
    "الدكتور كتبلي روشتة فيها مضاد حيوي ومسكن بعد الأكل.",
    "بدور على شقة مفروشة للإيجار في {loc} قريبة من المترو.",
    "صاحب الشقة طالب شهرين تأمين وإيجار {price} جنيه في الشهر.",
    "النهاردة الجو جميل أوي وفي هوا حلو على النيل.",
    "الجو حر موت في الصيف، شغل التكييف لو سمحت.",
    "صحيت بدري النهاردة على آذان الفجر وشربت {food}.",
    "عندي اجتماع مهم مع المدير الساعة {time} في {loc}.",
    "ألف مبروك على الشغل الجديد، ربنا يوفقك ويكرمك.",
]


def build_corpus(seed_path: Path, out_path: Path, target_count: int = 16000) -> list[str]:
    random.seed(42)
    corpus_list = []
    corpus_set = set()

    if seed_path.exists():
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not re.search(r"[A-Za-zÀ-ž]", clean) and re.search(r"[\u0600-\u06FF]", clean):
                norm = normalize_egyptian_text(clean)
                if 2 <= len(norm.split()) <= 20 and norm not in corpus_set:
                    corpus_set.add(norm)
                    corpus_list.append(norm)
        print(f"Loaded {len(corpus_list)} clean seed phrases from {seed_path}")

    for phrase in GREETINGS + LOANWORDS:
        norm = normalize_egyptian_text(phrase if phrase.endswith(".") else f"{phrase}.")
        if norm not in corpus_set:
            corpus_set.add(norm)
            corpus_list.append(norm)

    hours = list(range(1, 13))
    mins_opts = ["00", "15", "30", "45"]
    prices = [10, 15, 20, 25, 30, 40, 50, 75, 100, 120, 150, 200, 250, 300, 450, 500, 750, 1000, 1500, 2500, 3000, 5000]

    attempts = 0
    while len(corpus_list) < target_count and attempts < target_count * 25:
        attempts += 1
        sent = random.choice(TEMPLATES).format(
            loc=random.choice(LOCATIONS),
            food=random.choice(FOOD_DRINKS),
            price=random.choice(prices),
            time=f"{random.choice(hours)}:{random.choice(mins_opts)}",
        )
        if random.random() < 0.25:
            sent = f"{random.choice(GREETINGS)}، {sent}"
        if random.random() < 0.20 and sent.endswith("."):
            sent = sent[:-1] + "، " + random.choice(IDIOM_TAILS)

        norm = normalize_egyptian_text(sent)
        if 3 <= len(norm.split()) <= 18 and norm not in corpus_set:
            corpus_set.add(norm)
            corpus_list.append(norm)

    final_corpus = corpus_list[:target_count]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(final_corpus), encoding="utf-8")
    print(f"Saved {len(final_corpus)} unique Egyptian Arabic sentences to {out_path}")
    return final_corpus


# ==============================================================================
# 3. TEACHER DISTILLATION VIA VOICETUT-TTS (AHMED & MARIAM -> GOOGLE DRIVE)
# ==============================================================================
def run_distillation(corpus_path: Path, dataset_dir: Path):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for distillation. Switch Colab runtime to L4 or A100.")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/k2-fsa/OmniVoice.git", "voicetut-tts", "librosa", "soundfile"], check=True)

    import librosa
    import soundfile as sf
    from voicetut_tts import VoiceTutTTS

    wavs_dir = dataset_dir / "wavs"
    metadata_csv = dataset_dir / "metadata.csv"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    sentences = [l.strip() for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    speakers = [("Ahmed", "ahmed"), ("Mariam", "mariam")]

    # OHF-Voice/piper1-gpl multi-speaker CSV format: filename.wav|speaker_name|text
    completed = set()
    if metadata_csv.exists():
        with open(metadata_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) == 3 and (wavs_dir / row[0]).exists():
                    completed.add(row[0])

    print(f"Resuming distillation: {len(completed)}/{len(sentences) * 2} WAV clips already on Drive.")
    if len(completed) >= len(sentences) * 2:
        print("All WAV clips are already generated!")
        return

    tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")
    tmp_raw = Path("tmp_raw.wav")
    start_t = time.time()
    generated_now = 0

    with open(metadata_csv, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv, delimiter="|")

        for idx, text in enumerate(sentences, 1):
            prompt = text if text[-1] in ".!؟?،" else f"{text}."

            for teacher_name, spk_label in speakers:
                wav_name = f"{spk_label}_{idx:05d}.wav"
                out_wav = wavs_dir / wav_name

                if wav_name in completed and out_wav.exists() and out_wav.stat().st_size > 2000:
                    continue

                try:
                    tts.synthesize(
                        prompt,
                        speaker=teacher_name,
                        num_step=12,
                        guidance_scale=3.8,
                        speed=0.95,
                        output=str(tmp_raw),
                    )
                    audio, _ = librosa.load(str(tmp_raw), sr=22050, mono=True)
                    audio_trimmed, _ = librosa.effects.trim(audio, top_db=35)
                    sf.write(str(out_wav), audio_trimmed, 22050, subtype="PCM_16")

                    writer.writerow([wav_name, spk_label, text])
                    f_csv.flush()
                    generated_now += 1
                except Exception as e:
                    print(f"Error synthesizing {wav_name}: {e}")

            if idx % 100 == 0 or idx == len(sentences):
                elapsed = max(time.time() - start_t, 0.1)
                rate = generated_now / elapsed
                rem = (len(sentences) * 2) - (len(completed) + generated_now)
                eta_hrs = (rem / max(rate, 0.01)) / 3600
                print(f"[{idx}/{len(sentences)}] Clips: {len(completed) + generated_now}/{len(sentences)*2} | {rate:.2f} clips/s | ETA: {eta_hrs:.2f}h")


# ==============================================================================
# 4. PIPER1-GPL 2-SPEAKER VITS TRAINING & ONNX EXPORT
# ==============================================================================
def run_piper_training(dataset_dir: Path, max_epochs: int = 1200, batch_size: int = 32):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for Piper VITS training.")

    piper_dir = Path("/content/piper1-gpl")
    if not piper_dir.exists():
        print("Cloning and building OHF-Voice/piper1-gpl...")
        subprocess.run(["apt-get", "-qq", "install", "-y", "python3-dev", "cmake", "build-essential"], check=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/OHF-Voice/piper1-gpl.git", str(piper_dir)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-build-core", "scikit-build", "cython", "onnx", "onnxscript", "huggingface_hub"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{piper_dir}[train]"], check=True)
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(piper_dir), check=True)

    # Download Arabic ar_JO-kareem-medium checkpoint for vocoder warmstart
    from huggingface_hub import hf_hub_download
    warmstart_ckpt = hf_hub_download(
        repo_id="rhasspy/piper-checkpoints",
        repo_type="dataset",
        filename="ar/ar_JO/kareem/medium/epoch=4999-step=1364000.ckpt",
    )
    print(f"Downloaded vocoder warmstart checkpoint: {warmstart_ckpt}")

    work_dir = dataset_dir / "piper_training"
    cache_dir = work_dir / "cache"
    config_path = work_dir / "masri-arz-2spk.onnx.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if resuming from a previous training checkpoint on Drive
    existing_ckpts = sorted(glob.glob(str(work_dir / "lightning_logs" / "version_*" / "checkpoints" / "last.ckpt")))
    resume_args = ["--ckpt_path", existing_ckpts[-1]] if existing_ckpts else []
    if resume_args:
        print(f"Resuming training from checkpoint: {existing_ckpts[-1]}")

    train_cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.voice_name", "masri-arz-2spk",
        "--data.csv_path", str(dataset_dir / "metadata.csv"),
        "--data.audio_dir", str(dataset_dir / "wavs"),
        "--model.sample_rate", "22050",
        "--data.phoneme_type", "text",
        "--data.espeak_voice", "ar",
        "--data.cache_dir", str(cache_dir),
        "--data.config_path", str(config_path),
        "--data.batch_size", str(batch_size),
        "--data.validation_split", "0.02",
        "--data.num_test_examples", "10",
        "--model.vocoder_warmstart_ckpt", warmstart_ckpt,
        "--trainer.default_root_dir", str(work_dir),
        "--trainer.max_epochs", str(max_epochs),
        "--trainer.accelerator", "gpu",
        "--trainer.devices", "1",
        "--trainer.precision", "16-mixed",
        *resume_args,
    ]

    print("Starting Piper 2-Speaker VITS Training...")
    subprocess.run(train_cmd, check=True)

    # Locate best or last checkpoint and export to ONNX
    ckpts = sorted(glob.glob(str(work_dir / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found after training!")

    best_ckpt = next((c for c in ckpts if "last.ckpt" in c), ckpts[-1])
    export_dir = dataset_dir / "exported_model"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = export_dir / "masri-arz-2spk.onnx"
    json_out = export_dir / "masri-arz-2spk.onnx.json"

    print(f"Exporting {best_ckpt} -> {onnx_out} ...")
    subprocess.run([
        sys.executable, "-m", "piper.train.export_onnx",
        "--checkpoint", best_ckpt,
        "--output-file", str(onnx_out),
    ], check=True)
    shutil.copy2(config_path, json_out)
    print(f"\nDONE! Ready-to-deploy VPS/Android model saved on Google Drive:\n  - {onnx_out}\n  - {json_out}")


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Masri Egyptian Arabic 2-Speaker TTS Pipeline")
    parser.add_argument("--stage", choices=["corpus", "generate", "train", "all"], default="all")
    parser.add_argument("--target-count", type=int, default=16000)
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/masri_tts_dataset")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    seed_path = repo_root / "corpus" / "seed_phrases.txt"
    corpus_path = repo_root / "corpus" / "corpus_16k.txt"
    dataset_dir = Path(args.drive_dir)

    if args.stage in ("corpus", "all"):
        build_corpus(seed_path, corpus_path, args.target_count)

    if args.stage in ("generate", "all"):
        if not corpus_path.exists():
            build_corpus(seed_path, corpus_path, args.target_count)
        run_distillation(corpus_path, dataset_dir)

    if args.stage in ("train", "all"):
        run_piper_training(dataset_dir, max_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
