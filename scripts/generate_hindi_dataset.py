"""
generate_hindi_dataset.py
─────────────────────────
Generates a local Hindi audio dataset using macOS built-in
Lekha (hi_IN) TTS voice. No internet required.

Creates:
  data/hindi_tts/
    train/   — 200 audio files + train.csv
    validation/ — 50 audio files + validation.csv
    test/    — 50 audio files + test.csv

Run:
    python scripts/generate_hindi_dataset.py
"""

import os
import csv
import subprocess
import random
from pathlib import Path

# ── Hindi sentences ────────────────────────────────────────
# Common Hindi sentences covering daily topics
HINDI_SENTENCES = [
    "नमस्ते, आप कैसे हैं?",
    "मेरा नाम कुशल है।",
    "आज मौसम बहुत अच्छा है।",
    "मुझे चाय पीना पसंद है।",
    "यह किताब बहुत रोचक है।",
    "मैं दिल्ली में रहता हूँ।",
    "क्या आप हिंदी बोलते हैं?",
    "मुझे भूख लगी है।",
    "यह बाज़ार बहुत बड़ा है।",
    "कल मैं मुंबई जाऊँगा।",
    "भारत एक महान देश है।",
    "मेरे परिवार में चार लोग हैं।",
    "यह फिल्म बहुत अच्छी है।",
    "मैं रोज़ सुबह व्यायाम करता हूँ।",
    "आपका घर कहाँ है?",
    "मुझे संगीत बहुत पसंद है।",
    "यह खाना बहुत स्वादिष्ट है।",
    "मैं एक विद्यार्थी हूँ।",
    "कृपया धीरे बोलिए।",
    "धन्यवाद, आपकी बहुत कृपा है।",
    "मेरी माँ बहुत अच्छी हैं।",
    "आज रविवार है।",
    "मुझे पढ़ना बहुत पसंद है।",
    "यह रास्ता कहाँ जाता है?",
    "बाज़ार कितनी दूर है?",
    "मैं हर दिन स्कूल जाता हूँ।",
    "पानी बहुत ज़रूरी है।",
    "आसमान नीला है।",
    "फूल बहुत सुंदर हैं।",
    "मेरा पसंदीदा रंग नीला है।",
    "क्या आप चाय लेंगे?",
    "यह काम बहुत मुश्किल है।",
    "मैं कल वापस आऊँगा।",
    "आपसे मिलकर बहुत अच्छा लगा।",
    "यहाँ बहुत शोर है।",
    "मुझे नींद आ रही है।",
    "बच्चे खेल रहे हैं।",
    "यह दुकान बंद है।",
    "मौसम बदल रहा है।",
    "मेरे पास समय नहीं है।",
    "क्या आप मेरी मदद कर सकते हैं?",
    "यह सड़क बहुत व्यस्त है।",
    "आज बहुत गर्मी है।",
    "मैं थक गया हूँ।",
    "खिड़की खोल दीजिए।",
    "दरवाज़ा बंद कर दो।",
    "मुझे अस्पताल जाना है।",
    "डॉक्टर साहब कहाँ हैं?",
    "यह दवाई कब लेनी है?",
    "मेरा सिर दर्द कर रहा है।",
    "आज बाज़ार जाना है।",
    "दूध लाना मत भूलना।",
    "खाना तैयार है।",
    "थाली परोस दो।",
    "चावल और दाल बनाओ।",
    "सब्ज़ी बाज़ार से लाओ।",
    "आम बहुत मीठे हैं।",
    "यह मौसम बरसात का है।",
    "बादल घिर आए हैं।",
    "बारिश शुरू हो गई है।",
    "छाता लेकर जाओ।",
    "ट्रेन कितने बजे आएगी?",
    "टिकट कहाँ से मिलेगी?",
    "यह बस कहाँ जाएगी?",
    "मुझे स्टेशन जाना है।",
    "रिक्शा कितने का है?",
    "मीटर से चलोगे?",
    "यहाँ से कितनी दूर है?",
    "सीधे जाओ फिर बाएँ मुड़ो।",
    "वहाँ एक बड़ा पेड़ है।",
    "मंदिर कहाँ है?",
    "यह शहर बहुत पुराना है।",
    "इतिहास पढ़ना ज़रूरी है।",
    "विज्ञान बहुत रोचक है।",
    "गणित मेरा पसंदीदा विषय है।",
    "परीक्षा कल है।",
    "मैंने अच्छे से पढ़ाई की है।",
    "नतीजे कब आएँगे?",
    "मुझे अच्छे अंक मिले।",
    "मेरे दोस्त बहुत अच्छे हैं।",
    "हम साथ खेलते हैं।",
    "क्रिकेट मेरा पसंदीदा खेल है।",
    "भारत ने मैच जीता।",
    "खिलाड़ी बहुत अच्छे हैं।",
    "मैदान बहुत बड़ा है।",
    "दर्शक खुश हैं।",
    "यह एक यादगार पल है।",
    "हमें मेहनत करनी चाहिए।",
    "सफलता ज़रूर मिलेगी।",
    "हार से मत घबराओ।",
    "कोशिश जारी रखो।",
    "उम्मीद रखो।",
    "जीवन बहुत सुंदर है।",
    "प्रकृति की रक्षा करो।",
    "पेड़ लगाओ।",
    "पानी बचाओ।",
    "बिजली की बचत करो।",
    "स्वच्छता ज़रूरी है।",
    "हाथ धोना मत भूलो।",
    "स्वास्थ्य ही धन है।",
    "रोज़ फल खाओ।",
    "व्यायाम करो और स्वस्थ रहो।",
    "नींद पूरी लो।",
    "तनाव से दूर रहो।",
    "खुश रहो और मुस्कुराओ।",
    "दूसरों की मदद करो।",
    "बड़ों का सम्मान करो।",
    "बच्चों से प्यार करो।",
    "देश की सेवा करो।",
    "ईमानदारी सबसे अच्छी नीति है।",
    "सच बोलना चाहिए।",
    "वादा निभाना ज़रूरी है।",
    "समय की कद्र करो।",
    "आज का काम आज करो।",
    "कल पर मत टालो।",
    "किताबें पढ़ो।",
    "ज्ञान ही शक्ति है।",
    "शिक्षा सबका अधिकार है।",
    "बेटी पढ़ाओ, बेटी बचाओ।",
    "सब बराबर हैं।",
    "प्रेम और भाईचारा रखो।",
    "मिलकर काम करो।",
    "एकता में शक्ति है।",
    "भारत हमारी मातृभूमि है।",
    "हम सब भारतीय हैं।",
    "जय हिन्द।",
    "वंदे मातरम।",
    "यह मेरा घर है।",
    "यहाँ बहुत शांति है।",
    "रात को तारे दिखते हैं।",
    "चाँद बहुत सुंदर है।",
    "सूरज पूरब से उगता है।",
    "शाम को आसमान लाल होता है।",
    "पहाड़ बहुत ऊँचे हैं।",
    "नदी बह रही है।",
    "समुद्र बहुत गहरा है।",
    "जंगल में जानवर रहते हैं।",
    "शेर जंगल का राजा है।",
    "हाथी बहुत बड़ा है।",
    "मोर भारत का राष्ट्रीय पक्षी है।",
    "कमल भारत का राष्ट्रीय फूल है।",
    "आम भारत का राष्ट्रीय फल है।",
    "हॉकी भारत का राष्ट्रीय खेल है।",
    "तिरंगा हमारा राष्ट्रीय ध्वज है।",
    "जन गण मन हमारा राष्ट्रगान है।",
    "गांधीजी हमारे राष्ट्रपिता हैं।",
    "नेहरूजी पहले प्रधानमंत्री थे।",
    "भारत १९४७ में आज़ाद हुआ।",
    "संविधान हमारा मार्गदर्शक है।",
    "लोकतंत्र सबसे अच्छी व्यवस्था है।",
    "हर नागरिक को वोट देना चाहिए।",
    "कानून का पालन करो।",
    "अधिकार और कर्तव्य दोनों ज़रूरी हैं।",
    "न्याय सबको मिलना चाहिए।",
    "भ्रष्टाचार से लड़ो।",
    "देश को आगे बढ़ाओ।",
    "नई तकनीक अपनाओ।",
    "डिजिटल भारत बनाओ।",
    "स्टार्टअप शुरू करो।",
    "नवाचार ज़रूरी है।",
    "युवाओं में शक्ति है।",
    "भविष्य उज्जवल है।",
    "सपने देखो और पूरे करो।",
    "कड़ी मेहनत रंग लाती है।",
    "धैर्य रखो।",
    "हिम्मत मत हारो।",
    "आगे बढ़ते रहो।",
    "मंज़िल ज़रूर मिलेगी।",
    "यह रात भी गुज़र जाएगी।",
    "सुबह होगी।",
    "उम्मीद की किरण है।",
    "ज़िंदगी खूबसूरत है।",
    "हर पल को जियो।",
    "खुशियाँ बाँटो।",
    "प्यार फैलाओ।",
    "दुनिया को बेहतर बनाओ।",
    "यही हमारा लक्ष्य है।",
    "हम मिलकर कर सकते हैं।",
    "जय भारत।",
]


def generate_audio(text: str, output_path: str, voice: str = "Lekha") -> bool:
    """Generate audio using macOS say command."""
    try:
        # Generate AIFF first (macOS say only outputs AIFF natively)
        aiff_path = output_path.replace(".wav", ".aiff")
        subprocess.run(
            ["say", "-v", voice, "-o", aiff_path, text],
            check=True,
            capture_output=True,
        )
        # Convert to WAV using afconvert (built into macOS)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff_path, output_path],
            check=True,
            capture_output=True,
        )
        # Remove temp AIFF
        os.remove(aiff_path)
        return True
    except Exception as e:
        print(f"  Error generating audio: {e}")
        return False


def create_split(sentences, split_name, base_dir):
    """Generate audio files and CSV for a dataset split."""
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\nGenerating {split_name} split ({len(sentences)} samples)...")

    for i, sentence in enumerate(sentences):
        audio_filename = f"{split_name}_{i:04d}.wav"
        audio_path = split_dir / audio_filename

        success = generate_audio(sentence, str(audio_path))
        if success:
            rows.append({
                "path": str(audio_path),
                "sentence": sentence,
                "split": split_name,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sentences)} done...")

    # Save CSV
    csv_path = split_dir / f"{split_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} samples to {split_dir}")
    return csv_path


def main():
    base_dir = Path("data/hindi_tts")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle for variety
    sentences = HINDI_SENTENCES.copy()
    random.seed(42)
    random.shuffle(sentences)

    # Split into train / validation / test
    # Use all 200 sentences — repeat if needed
    all_sentences = sentences * 2  # 300+ sentences available
    train_sentences      = all_sentences[:200]
    validation_sentences = all_sentences[200:250]
    test_sentences       = all_sentences[250:300]

    train_csv = create_split(train_sentences, "train", base_dir)
    val_csv   = create_split(validation_sentences, "validation", base_dir)
    test_csv  = create_split(test_sentences, "test", base_dir)

    print(f"\nDataset ready at: {base_dir}")
    print(f"  Train:      {train_csv}")
    print(f"  Validation: {val_csv}")
    print(f"  Test:       {test_csv}")
    print("\nRun training with:")
    print("  PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --smoke_test")


if __name__ == "__main__":
    main()