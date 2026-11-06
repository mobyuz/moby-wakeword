# Upstream asboblar zanjiri hisoboti (W0)

Sana: 2026-11-05. Mashina: Windows 11 Pro 10.0.26200, Python 3.12.10, faqat CPU, diskda D:.

**Xulosa.** microWakeWord'ning haqiqiy quvuri (xususiyat generatsiyasi, o'qitish, oqimli modelga aylantirish, int8 kvantlash) shu Windows mashinasida **native** ishladi, WSL2'siz, kichik sintetik namunada. Bu faqat asboblar zanjirining ishlashi isboti. Namuna nutq emas (uch ohangli naqsh va shovqin), olingan 27 936 baytlik model nutq modeli emas va hech qanday aniqlik da'vosiga asos bo'lmaydi.

## Birlamchi manbalardan topilgan (OHF-Voice/micro-wake-word, commit `4665173`, 2026-07-06)

- Litsenziya: Apache-2.0 (repo `LICENSE`, `pyproject.toml` classifier).
- Python: `requires-python >=3.10`; notebook "You should use Python 3.10". `setup.py`: `tensorflow>=2.18`, `numpy>=2.0`, `ai-edge-litert`, `pymicro-features`, `webrtcvad-wheels`, `audiomentations`, `mmap_ninja`, `datasets`, `audio_metadata`, `pyyaml`.
- O'rnatish: PyPI paketi yo'q; `git clone` + `pip install -e`. Relizlar bo'sh, versiya 0.1.0.
- README o'zi: "currently available as an early release. Training new models is intended for advanced users." Va notebook model "most likely not be usable" bo'lishini aytadi.
- Oqim: Piper sample generator bilan TTS, spektrogramma xususiyatlari (40 ta, 10 ms qadam) Ragged Mmap'da, nostreaming o'qitish, keyin oqimli va kvantlangan TFLite. Chiqish: `trained_models/<nom>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`.
- Windows/WSL/Colab haqida upstream hech narsa yozmaydi. Notebook Colab'da ishlaydi ("extremely slow", GPU kerak), mahalliy GPU tavsiya qilinadi. macOS uchun fork'lar ko'rsatilgan, Windows uchun yo'q.

## Bu mashinada nima tekshirildi

| Qadam | Natija |
| --- | --- |
| `pip install tensorflow==2.20.0` + upstream (`-e`) Python 3.12 Windows'da | o'rnatildi; `pip check`: buzilgan bog'liqlik yo'q |
| `webrtcvad-wheels`, `pymicro-features`, `ai-edge-litert` Windows wheel | mavjud, o'rnatildi |
| `Clips`, `Augmentation`, `SpectrogramGeneration`, `RaggedMmap` | ishladi (augmentatsiya: rang shovqini va gain; fon shovqini va RIR o'chirilgan: yuklab olish yo'q) |
| `microwakeword.model_train_eval ... mixednet` 40 qadam, kichik tarmoq (5 601 parametr) | o'qitildi, oqimli kvantlangan TFLite hosil bo'ldi |
| Natija fayli | `stream_state_internal_quant.tflite`, 27 936 bayt, SHA-256 `3a0b6ae3b352a5f6870b9d38a7e0c714cf010b9390248e9933320fae8f00e4e2` (D: dagi scratch'da, git'da emas) |

Yuklab olingan hajm: pip keshi ~620 MB (D:); `.venv-train` ~2.1 GB disk hajmi, smoke scratch ~124 MB. TTS, Piper modeli, salbiy to'plamlar, RIR **yuklanmadi**.

### Nimani aniqladik (tuzatish kerak bo'lgan xatolar)

1. **`datasets` 4+ ishlamaydi.** Upstream `Clips` audioni `datasets.Audio` orqali dekodlaydi; `datasets` 5.1.0 `torchcodec` talab qildi (`ImportError: To support decoding audio data, please install 'torchcodec'`). `datasets==3.6.0` ga tushirish ishladi. `requirements-train.txt` shuni qadaydi.
2. **Birinchi chaqiruvlar juda sekin.** Sovuq `import tensorflow` bir daqiqadan oshdi, birinchi spektrogramma ~90 s (JIT/kompilyatsiya), keyingilari tez. Bu "osilib qolish" emas; faqat kutish kerak.
3. **PyPI `audio-metadata` 0.11.1** ishladi; notebook'dagi fork kerak bo'lmadi.
4. `ai-edge-litert`, `numpy 2.5.4`, `keras 3.15.1` bilan ishladi; upstream chegarasi `numpy>=2.0`, `tensorflow>=2.18`.
5. Tezlik: o'lchanmagan. 40 qadam CPU'da bir necha daqiqa; to'liq o'qitish (10 000+ qadam, katta ma'lumot) uchun CPU vaqti yoki GPU kerak va hozircha da'vo qilinmaydi.

### Ishlamagan yoki sinalmagan

- **WSL2:** `Ubuntu` taqsimoti o'rnatilgan, lekin to'xtagan; C: to'la bo'lgani uchun (WSL diski C: da) ishga tushirilmadi. Windows native ishlagani uchun kerak bo'lmadi. GPU o'qitish uchun WSL2 yoki Linux ehtimol kerak (TensorFlow 2.11 dan keyin native Windows GPU qo'llab-quvvatlamaydi; bu ma'lumot tekshirilmadi, shu mashinada GPU sinalmadi).
- **Piper namuna generatori** (sun'iy talaffuz): yuklanmadi; PyTorch va ovoz modeli kerak, ovoz modeli litsenziyasi aniq emas ([licenses.md](licenses.md)). Isbotlovchi buyruq (Linux/WSL2 yoki Colab'da): upstream notebook'ning 3-4-hujayralari, `python3 piper-sample-generator/generate_samples.py "khum_puter" --max-samples 1 --batch-size 1 --output-dir generated_samples`.
- **Colab:** sinalmadi.
- **Linux CI'da smoke:** sinalmadi; `ci.yml` da `continue-on-error` bilan turibdi.
- **Haqiqiy nutq:** yo'q.

## Qayta ishga tushirish

```powershell
$env:TEMP='D:\moby-cache\tmp'; $env:TMP='D:\moby-cache\tmp'; $env:PIP_CACHE_DIR='D:\moby-cache\pip'
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv-train\Scripts\python.exe -u scripts\upstream_smoke.py --workdir D:\moby-cache\smoke --steps 40
```

Skript oxirida `OK: <fayl> <bayt> sha256=...` yoki `FAILED` chiqaradi. Skript papkasiga stdlib nomli fayl (`inspect.py` kabi) qo'ymang: Python skript papkasini qidiruv yo'liga birinchi qo'yadi.

## Versiyalar (`.venv-train`)

Python 3.12.10; tensorflow 2.20.0; keras 3.15.1; numpy 2.5.4; ai-edge-litert 2.3.0; pymicro-features 2.0.2; microwakeword 0.1.0 (commit `4665173cd35f1cff9a61e06fc427f124766c488e`); audiomentations 0.43.1; mmap_ninja 0.9.0; webrtcvad-wheels 2.0.14.post1; audio-metadata 0.11.1; datasets 3.6.0. To'liq ro'yxat: [`requirements-train.txt`](../requirements-train.txt). Tooling `.venv`: ruff 0.17.3, mypy 2.4.1, pytest 9.2.0, PyYAML 6.0.3.

## W2: Colab daftari (SINALMAGAN, UNTESTED)

Fayllar: [`scripts/colab/moby_wakeword_colab.py`](../scripts/colab/moby_wakeword_colab.py) (`# %%` hujayralari; `scripts/colab/make_ipynb.py` uni `.ipynb` ga aylantiradi), `synthesize_piper.py`, `mww_steps.py`, `train_seeded.py`. **Colab'da hech qachon ishga tushirilmagan.** O'qitish faqat Colab'da (GPU); mahalliy mashinada o'qitish yoki TTS generatsiyasi yo'q.

Oqim: pinlangan `requirements-train.txt` (microWakeWord commit `4665173`) va repo -> `python -m mobyww.synth plan` -> Piper (`piper-tts==1.3.0`, alohida virtualenv, `piper-sample-generator` commit `2971426a`) ovozlari bilan TTS -> 16 kHz -> manifest -> MUSAN (noise, speech) va ixtiyoriy BUT ReverbDB RIR rasmiy URL'lardan (hajm + MD5/SHA-256) -> `mobyww.augment` (qurilma yo'li bilan) -> upstream xususiyatlari -> o'qitish (seed bilan) -> `provenance.json` modelning yoniga (repo va upstream commit, urug'lar, giperparametrlar, manifest xeshi).

Maxfiylik: daftarda yuklash (`files.upload`) va Drive ulash yo'q; `recordings` va `consent` papkalaridan yo'l rad etiladi; haqiqiy yozuvlar (~20 min) lokal qoladi va faqat test to'plami bo'ladi. `tests/test_colab_guard.py` buni tekshiradi (static qidiruv + guard hujayrasini ishga tushirish).

### Nima mahalliy tekshirilgan, nima yo'q

| Qadam | Holat |
| --- | --- |
| Augmentatsiya, qurilma yo'li, planner, manifest yozuvchi, fetch skripti, konfig va provenance yordamchilari | mahalliy birlik testlari (Windows, CPU) |
| Fetch: haqiqiy `read_me.txt` (14 529 bayt) yuklash, uzilgan `.part` dan davom ettirish, pinlangan SHA-256 | mahalliy ishga tushirildi |
| `mww_steps.build_features` va `mww_steps.train` (`train_seeded.py` orqali) W0 sinus-ohang namunasida (60/10/10 bo'lish papkalari, 40 qadam, 16 filtrli kichik tarmoq, `--seed 4242`) | mahalliy ishga tushirildi (Windows, CPU, 3 min 31 s): chiqish kodi 0, `stream_state_internal_quant.tflite` 27 936 bayt hosil bo'ldi. Namuna nutq emas, upstream chiqargan AUC/FRR raqamlari ma'nosiz va keltirilmaydi |
| Piper bilan TTS, ovoz yuklash va xesh tekshiruvi (`synthesize_piper.py`) | **sinalmagan** |
| MUSAN (11 GB) va ReverbDB (9.3 GB) to'liq yuklash va ochish | **sinalmagan** (faqat registr, dry-run va soxta server testlari) |
| Colab'da `requirements-train.txt` (Windows'dan muzlatilgan) o'rnatish, GPU, TensorFlow 2.20 CUDA | **sinalmagan**; GPU ko'rinmasa daftar to'xtaydi |
| Butun daftar boshidan oxirigacha | **sinalmagan** |

Ma'lum cheklovlar: Piper ONNX ovozlari urug' qabul qilmaydi, shuning uchun generatsiya qilingan audio qayta tiklanmaydi (job fayli, ovoz xeshlari va manifest xeshi qayd etiladi); GPU yadrolari ham aniq deterministik emas. Sintetik val/test optimistik (`docs/evaluation.md`); daftar aniqlik o'lchamaydi.
