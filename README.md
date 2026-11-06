# Moby wake word

Robot uyg'otuvchi so'zni ("Hey Moby") o'zi, ESP32-S3 ustida eshitishi kerak: ovoz u uyg'ongunga qadar robotdan tashqariga chiqmaydi. Bu repozitoriy [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) bilan model o'qitish, baholash va eksport qilish uchun. Robot kodi bu yerda yo'q.

**Holat: W0 va W1 tugagan, W2 kodi yozilgan (qisman). Hech qanday model o'qitilmagan, wake word aniqligi o'lchanmagan.** Upstream asboblar zanjirining kichik sintetik namunada ishlashi isbotlangan ([hisobot](docs/upstream-toolchain.md)), lekin bu nutq modeli emas. W2 da yozildi va birlik testlaridan o'tdi: augmentatsiya (INMP441 qurilma yo'li firmware konversiyasi bilan bit-bit mos), sintetik nutq rejalovchisi va manifest yozuvchi, ochiq ma'lumot yuklagich. Colab daftari yozilgan, lekin **sinalmagan**: Colab'da hech qachon ishga tushirilmagan, TTS generatsiya qilinmagan, ochiq to'plamlar (MUSAN, ReverbDB) to'liq yuklanmagan. Ibora `hey moby` (egasi 2026-11-05 da tanladi). O'tish chegaralari egasi tomonidan tasdiqlangan ([targets.yaml](configs/targets.yaml)); rozilik matni tasdig'i hali kutilmoqda. Haqiqiy yozuvlar (~20 min) lokal qoladi va faqat test to'plami bo'ladi.

## Nima bor

| Yo'l | Mazmuni |
| --- | --- |
| `configs/` | `wakeword.yaml` (ibora, qat'iy salbiylar, urug'lar), `targets.yaml` (egasi tasdiqlagan chegaralar), `augmentation.yaml` (W2 augmentatsiya rejasi va qurilma yo'li) |
| `src/mobyww/data/` | manifest qatori, validator, ovoz egasi bo'yicha bo'lish, kichik sintetik namuna |
| `src/mobyww/recording/` | yo'naltiruvchi yozish CLI'si, lokal rozilik yozuvlari, almashtiriladigan audio manba |
| `src/mobyww/guard.py` | yozuv, rozilik, shaxs va og'irliklarni git'dan tashqarida ushlovchi tekshiruv |
| `src/mobyww/augment/` | SNR aralashtirish, RIR, gain, vaqt siljishi va INMP441 qurilma yo'li (`moby_pcm.c` ga bit-bit mos) |
| `src/mobyww/synth/` | sintetik nutq rejasi (Colab job fayli) va hosil bo'lgan WAV'lardan manifest; generatsiya yo'q |
| `src/mobyww/train/` | o'qitish konfigi, split papkalari va `provenance.json` yordamchilari (TensorFlow'siz) |
| `src/mobyww/{evaluate,export,replay}/` | keyingi bosqichlar uchun bo'sh o'rinlar (W4 va undan keyin) |
| `scripts/` | `fetch_open_data.py` (faqat litsenziyasi ruxsat etilgan ochiq to'plamlar), `colab/` (Colab daftari, SINALMAGAN), `upstream_smoke.py` |
| `docs/` | [data-collection](docs/data-collection.md), [privacy](docs/privacy.md), [licenses](docs/licenses.md), [ATTRIBUTION](docs/ATTRIBUTION.md), [evaluation](docs/evaluation.md), [integration-contract](docs/integration-contract.md), [upstream-toolchain](docs/upstream-toolchain.md), [ADR](docs/adr) |
| `data/` | git'da emas; faqat [README](data/README.md) va [manifest sxemasi](data/manifest.schema.json) |

## O'rnatish (Windows 10/11, PowerShell)

Kerak: Git, Python 3.12. [just](https://github.com/casey/just) ixtiyoriy; yo'q bo'lsa quyidagi buyruqlar bevosita ishlaydi.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
git config core.hooksPath .githooks
```

Tekshiruvlar (`just check` ning ekvivalenti):

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m mobyww.data.sample
.\.venv\Scripts\python.exe -m mobyww.guard --tracked
```

Disk to'la bo'lsa venv, pip keshi va yuklamalarni boshqa diskka qo'ying (`$env:PIP_CACHE_DIR`, `$env:TEMP`).

O'qitish muhiti alohida (`.venv-train`, TensorFlow bilan) va CI'dagi yengil muhitga aralashmaydi. Buyruqlar va aniqlangan cheklovlar: [docs/upstream-toolchain.md](docs/upstream-toolchain.md).

## Yozish va tekshirish

```powershell
python -m mobyww.recording.cli plan                      # yozmasdan reja
python -m mobyww.recording.cli consent                   # rozilik, lokal
python -m mobyww.recording.cli session --speaker spk-xxxxxx --language uz --room kitchen --device usb-mic
python -m mobyww.data.validate data/manifests/recorded.jsonl --root data --consent-dir data/consent --allow-unassigned
```

Yozish uchun `pip install -e ".[record]"` (sounddevice) kerak. Tafsilot: [docs/data-collection.md](docs/data-collection.md).

## Litsenziya

GPL-3.0-only, [LICENSE](LICENSE). Uchinchi tomon ma'lumotlari va og'irliklari litsenziyasi [docs/licenses.md](docs/licenses.md) da; aniq bo'lmaganlari "stopped, not used" deb belgilangan.
