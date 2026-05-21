"""
record_voice.py
───────────────
Records your own Hindi voice for ASR training data.

How it works:
  - Shows you one Hindi sentence at a time
  - Press ENTER to start recording
  - Speak the sentence clearly
  - Recording stops automatically after silence
  - Saves audio + transcript to data/my_voice/

Run:
    python scripts/record_voice.py

Output:
    data/my_voice/train/   — 70% of recordings
    data/my_voice/val/     — 15% of recordings
    data/my_voice/test/    — 15% of recordings
    data/my_voice/recordings.csv — full transcript list
"""

import os
import csv
import time
import random
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Settings ────────────────────────────────────────
SAMPLE_RATE     = 16000   # Whisper expects 16kHz
SILENCE_THRESH  = 0.01    # Volume below this = silence
SILENCE_SECS    = 1.5     # Stop after this many seconds of silence
MAX_DURATION    = 15      # Max recording length in seconds
OUTPUT_DIR      = Path("data/my_voice")

# ── Hindi sentences to record ────────────────────────
# 150 sentences across daily topics
# Read each one clearly and naturally — don't rush
SENTENCES = [
    # Greetings & introductions
    "नमस्ते, आप कैसे हैं?",
    "मेरा नाम कुशल है।",
    "मैं एक विद्यार्थी हूँ।",
    "आपसे मिलकर बहुत खुशी हुई।",
    "मैं मुंबई से हूँ।",
    "आप कहाँ रहते हैं?",
    "मेरे परिवार में चार लोग हैं।",
    "मेरी माँ बहुत अच्छी हैं।",
    "मेरे पिताजी डॉक्टर हैं।",
    "मेरी एक छोटी बहन है।",

    # Daily life
    "आज मौसम बहुत अच्छा है।",
    "मुझे चाय पीना पसंद है।",
    "खाना तैयार है, आइए खाते हैं।",
    "मुझे भूख लगी है।",
    "मुझे नींद आ रही है।",
    "आज बहुत थकान हुई।",
    "कल मैं जल्दी उठूँगा।",
    "मैं रोज़ सुबह व्यायाम करता हूँ।",
    "दूध लाना मत भूलना।",
    "बाज़ार से सब्ज़ी लाओ।",

    # Food
    "यह खाना बहुत स्वादिष्ट है।",
    "आम बहुत मीठे हैं।",
    "चावल और दाल बनाओ।",
    "मुझे मिठाई बहुत पसंद है।",
    "क्या आप चाय लेंगे?",
    "यह सब्ज़ी बहुत अच्छी बनी है।",
    "रोटी गर्म करके खाओ।",
    "पानी बहुत ज़रूरी है।",
    "फल खाना सेहत के लिए अच्छा है।",
    "आज रसोई में क्या बना है?",

    # Education
    "मुझे पढ़ना बहुत पसंद है।",
    "परीक्षा कल है।",
    "मैंने अच्छे से पढ़ाई की है।",
    "गणित मेरा पसंदीदा विषय है।",
    "विज्ञान बहुत रोचक है।",
    "किताबें पढ़ो, ज्ञान बढ़ाओ।",
    "शिक्षा सबका अधिकार है।",
    "नतीजे कब आएँगे?",
    "मुझे अच्छे अंक मिले।",
    "मेरे शिक्षक बहुत अच्छे हैं।",

    # Travel & directions
    "यह रास्ता कहाँ जाता है?",
    "बाज़ार कितनी दूर है?",
    "सीधे जाओ फिर बाएँ मुड़ो।",
    "ट्रेन कितने बजे आएगी?",
    "टिकट कहाँ से मिलेगी?",
    "मुझे स्टेशन जाना है।",
    "रिक्शा कितने का है?",
    "कल मैं मुंबई जाऊँगा।",
    "यात्रा सुखद रही।",
    "हवाई अड्डा यहाँ से दूर है।",

    # Weather & nature
    "आज बहुत गर्मी है।",
    "बारिश शुरू हो गई है।",
    "छाता लेकर जाओ।",
    "मौसम बदल रहा है।",
    "आसमान नीला है।",
    "बादल घिर आए हैं।",
    "रात को तारे दिखते हैं।",
    "चाँद बहुत सुंदर है।",
    "सूरज पूरब से उगता है।",
    "आज ठंड बहुत है।",

    # Health
    "मेरा सिर दर्द कर रहा है।",
    "मुझे अस्पताल जाना है।",
    "डॉक्टर साहब कहाँ हैं?",
    "यह दवाई कब लेनी है?",
    "स्वास्थ्य ही धन है।",
    "रोज़ फल खाओ।",
    "नींद पूरी लो।",
    "तनाव से दूर रहो।",
    "हाथ धोना मत भूलो।",
    "व्यायाम करो और स्वस्थ रहो।",

    # Sports & entertainment
    "क्रिकेट मेरा पसंदीदा खेल है।",
    "भारत ने मैच जीता।",
    "यह फिल्म बहुत अच्छी है।",
    "मुझे संगीत बहुत पसंद है।",
    "खिलाड़ी बहुत अच्छे हैं।",
    "दर्शक बहुत खुश हैं।",
    "आज मैदान में अच्छा खेल हुआ।",
    "मुझे तैरना पसंद है।",
    "बच्चे खेल रहे हैं।",
    "यह एक यादगार पल है।",

    # Technology
    "नई तकनीक अपनाओ।",
    "डिजिटल भारत बनाओ।",
    "मोबाइल फोन बहुत ज़रूरी है।",
    "इंटरनेट ने दुनिया बदल दी है।",
    "कंप्यूटर सीखना ज़रूरी है।",
    "आर्टिफिशियल इंटेलिजेंस का युग है।",
    "ऑनलाइन पढ़ाई बहुत आसान है।",
    "डेटा बहुत महत्वपूर्ण है।",
    "सोशल मीडिया का सही उपयोग करो।",
    "साइबर सुरक्षा का ध्यान रखो।",

    # Values & motivation
    "हमें मेहनत करनी चाहिए।",
    "सफलता ज़रूर मिलेगी।",
    "हार से मत घबराओ।",
    "कोशिश जारी रखो।",
    "ईमानदारी सबसे अच्छी नीति है।",
    "सच बोलना चाहिए।",
    "समय की कद्र करो।",
    "आज का काम आज करो।",
    "दूसरों की मदद करो।",
    "खुश रहो और मुस्कुराओ।",

    # India & culture
    "भारत एक महान देश है।",
    "हम सब भारतीय हैं।",
    "तिरंगा हमारा राष्ट्रीय ध्वज है।",
    "भारत १९४७ में आज़ाद हुआ।",
    "गांधीजी हमारे राष्ट्रपिता हैं।",
    "मोर भारत का राष्ट्रीय पक्षी है।",
    "आम भारत का राष्ट्रीय फल है।",
    "हिंदी हमारी राजभाषा है।",
    "भारत की संस्कृति बहुत समृद्ध है।",
    "विविधता में एकता हमारी पहचान है।",

    # Environment
    "पेड़ लगाओ, पर्यावरण बचाओ।",
    "पानी बचाओ।",
    "बिजली की बचत करो।",
    "प्लास्टिक का उपयोग कम करो।",
    "स्वच्छता ज़रूरी है।",
    "नदी को प्रदूषित मत करो।",
    "जंगल हमारी धरोहर है।",
    "सौर ऊर्जा का उपयोग करो।",
    "धरती को बचाना हमारी ज़िम्मेदारी है।",
    "हरियाली फैलाओ।",

    # Shopping & money
    "यह दुकान बंद है।",
    "कितने का है यह?",
    "थोड़ा सस्ता करो।",
    "मुझे यह नहीं चाहिए।",
    "पैसे ATM से निकालने हैं।",
    "यह बहुत महंगा है।",
    "मुझे रसीद दीजिए।",
    "ऑनलाइन पेमेंट कर देता हूँ।",
    "बचत करना ज़रूरी है।",
    "यह सामान अच्छी गुणवत्ता का है।",

    # Miscellaneous
    "कृपया धीरे बोलिए।",
    "क्या आप मेरी मदद कर सकते हैं?",
    "मुझे समझ नहीं आया।",
    "फिर से बोलिए।",
    "धन्यवाद, आपकी बहुत कृपा है।",
    "माफ़ कीजिए, मुझसे गलती हुई।",
    "ज़िंदगी खूबसूरत है।",
    "हर पल को जियो।",
    "मंज़िल ज़रूर मिलेगी।",
    "जय हिन्द।",
]


# ── Recording logic ──────────────────────────────────
def record_until_silence() -> np.ndarray:
    """
    Records audio until silence is detected for SILENCE_SECS seconds.
    Returns the recorded waveform as a numpy array.
    """
    print("  🎙  Recording... speak now!")
    chunks = []
    silent_chunks = 0
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    max_chunks = int(MAX_DURATION / 0.1)
    silence_limit = int(SILENCE_SECS / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            chunks.append(chunk.copy())
            volume = np.abs(chunk).mean()

            if volume < SILENCE_THRESH:
                silent_chunks += 1
                if silent_chunks >= silence_limit and len(chunks) > silence_limit + 5:
                    break
            else:
                silent_chunks = 0

    waveform = np.concatenate(chunks, axis=0).flatten()
    # Trim trailing silence
    waveform = waveform[:len(waveform) - int(SILENCE_SECS * SAMPLE_RATE)]
    return waveform


def save_audio(waveform: np.ndarray, path: Path):
    """Normalise and save waveform as WAV."""
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    sf.write(str(path), waveform, SAMPLE_RATE)


# ── Main recording session ───────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = OUTPUT_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    print("\n" + "═" * 55)
    print("  Hindi Voice Recording — ASR Training Data")
    print("═" * 55)
    print(f"  Total sentences: {len(SENTENCES)}")
    print(f"  Output folder:   {OUTPUT_DIR}")
    print("\n  Tips for best results:")
    print("  → Speak clearly at normal pace")
    print("  → Keep 20–30cm from your mic")
    print("  → Quiet room, no background noise")
    print("  → Recording stops automatically after silence")
    print("\n  Press ENTER to start each sentence")
    print("  Type 's' + ENTER to skip a sentence")
    print("  Type 'q' + ENTER to stop and save")
    print("═" * 55 + "\n")

    input("  Press ENTER when you're ready to begin...")

    recordings = []
    sentences = SENTENCES.copy()
    random.seed(42)
    random.shuffle(sentences)

    for i, sentence in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}]")
        print(f"  Say this: {sentence}")

        choice = input("  ENTER=record  s=skip  q=quit  → ").strip().lower()

        if choice == 'q':
            print("\n  Stopping early and saving...")
            break
        elif choice == 's':
            print("  Skipped.")
            continue

        # Record
        waveform = record_until_silence()
        duration = len(waveform) / SAMPLE_RATE

        if duration < 0.5:
            print("  Too short — skipped. Try again.")
            continue

        # Save audio
        filename = f"recording_{i:04d}.wav"
        audio_path = audio_dir / filename
        save_audio(waveform, audio_path)

        recordings.append({
            "path": str(audio_path),
            "sentence": sentence,
            "duration": round(duration, 2),
        })

        print(f"  Saved ({duration:.1f}s) — {len(recordings)} recorded so far")

    if not recordings:
        print("\n  No recordings made. Exiting.")
        return

    # ── Split into train / val / test ─────────────────
    random.shuffle(recordings)
    n = len(recordings)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    splits = {
        "train": recordings[:n_train],
        "val":   recordings[n_train:n_train + n_val],
        "test":  recordings[n_train + n_val:],
    }

    for split_name, split_data in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)

        csv_path = split_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "sentence", "duration"])
            writer.writeheader()
            writer.writerows(split_data)

    # ── Full CSV ──────────────────────────────────────
    full_csv = OUTPUT_DIR / "recordings.csv"
    with open(full_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "duration"])
        writer.writeheader()
        writer.writerows(recordings)

    # ── Summary ───────────────────────────────────────
    total_duration = sum(r["duration"] for r in recordings)
    print("\n" + "═" * 55)
    print(f"  Done! Recorded {len(recordings)} sentences")
    print(f"  Total audio:  {total_duration/60:.1f} minutes")
    print(f"  Train:        {len(splits['train'])} samples")
    print(f"  Validation:   {len(splits['val'])} samples")
    print(f"  Test:         {len(splits['test'])} samples")
    print(f"  Saved to:     {OUTPUT_DIR}")
    print("═" * 55)
    print("\n  To retrain with your voice data, update")
    print("  configs/config.yaml:")
    print("    local_data_dir: 'data/my_voice'")
    print("  Then run:")
    print("    python train.py --config configs/config.yaml")
    print()


if __name__ == "__main__":
    main()
