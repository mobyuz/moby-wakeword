# Litsenziyalar

Tekshirilgan sana: 2026-11-05. Holat ustuni: **allowed** (ishlatilishi mumkin), **candidate** (yo'l bor, lekin yuklab olishdan oldin birlamchi manbada tasdiqlanishi kerak), **stopped, not used** (aniq emas yoki cheklov bor; ishlatilmaydi). "Tarqatish" ustuni: o'qitilgan modelni (GPL-3.0-only repodan GitHub Release orqali) tarqatish mumkinmi.

Manba turi: **P** = birlamchi (rasmiy repo/sahifa/paket metadata'si shu sessiyada o'qildi), **S** = ikkilamchi (boshqa hujjat bu haqda yozgan, birlamchi tasdiqlanmadi). Hech narsa litsenziya matnining yuridik tahlili emas.

## Upstream kutubxonalar (kod; model tarqatishga ta'sir qilmaydi)

| Nom | Versiya | Litsenziya | Manba | Holat |
| --- | --- | --- | --- | --- |
| microWakeWord (`OHF-Voice/micro-wake-word`) | commit `4665173` (2026-07-06), paket 0.1.0 | Apache-2.0 | P: repo LICENSE fayli va `pyproject.toml` | allowed |
| TensorFlow | 2.20.0 | Apache-2.0 | P: paket metadata'si | allowed |
| Keras | 3.15.1 | Apache-2.0 | P: paket metadata'si | allowed |
| ai-edge-litert | 2.3.0 | Apache-2.0 (classifier) | P: paket metadata'si | allowed |
| pymicro-features | 2.0.2 | Apache | P: paket metadata'si (ichidagi kissfft va microfrontend sarlavhalari tekshirilmadi) | allowed, tafsilot tekshiruvi ochiq |
| audiomentations | 0.43.1 | MIT | P: paket metadata'si | allowed |
| mmap_ninja | 0.9.0 | MIT | P: paket metadata'si | allowed |
| webrtcvad-wheels | 2.0.14.post1 | MIT | P: paket metadata'si | allowed |
| audio-metadata | 0.11.1 (PyPI) | MIT | P: paket metadata'si | allowed |
| datasets (Hugging Face) | 3.6.0 | Apache-2.0 | P: paket metadata'si | allowed |
| NumPy, SciPy, PyYAML | 2.5.4, 1.18.1, 6.0.3 | BSD-3-Clause va boshqalar, BSD, MIT | P: paket metadata'si | allowed |

Upstream notebook'i `audio-metadata` ning ayrim fork'ini (`git+https://github.com/whatsnowplaying/audio-metadata@d4ebb23...`) va macOS uchun `pymicro-features` fork'ini o'rnatadi. Biz PyPI versiyasini ishlatdik (Windows, 3.12); fork'lar ishlatilmagan.

Loyihaning o'zi GPL-3.0-only; Apache-2.0 kodni GPL-3.0 loyihada ishlatish mumkin (FSF: Apache-2.0 GPLv3 bilan mos). Bu yuridik maslahat emas.

## TTS dvigatellari (W2 uchun nomzodlar; hali ishlatilmagan)

| Nom | Litsenziya | Manba | Tarqatish | Holat |
| --- | --- | --- | --- | --- |
| `rhasspy/piper-sample-generator` (kod) | MIT | P: repo sahifasi | kodga taalluqli emas | candidate |
| `en_US-libritts_r-medium.pt` (ovoz modeli, notebook yuklaydi) | repo va v2.0.0 reliz sahifasida **ko'rsatilmagan**; nomi LibriTTS-R ga ishora qiladi | P: litsenziya yo'q | noma'lum | **stopped, not used** |
| `piper-phonemize-cross` / espeak-ng fonemalash | tekshirilmadi | yo'q | noma'lum | **stopped, not used** |
| O'zbek va rus TTS ovozlari | tanlanmagan; piper-sample-generator ovozi inglizcha (README: faqat ingliz) | P | noma'lum | **stopped, not used** |

Reja talabi: har dvigatelning litsenziyasi va generatsiya qilingan audioni o'qitishda ishlatish ruxsati W2 oldidan alohida tekshiriladi.

## Salbiy va fon ma'lumotlari

| Nom | Litsenziya | Manba | Tarqatish | Holat |
| --- | --- | --- | --- | --- |
| Mozilla Common Voice, o'zbek (Common Voice Scripted Speech 25.0 - Uzbek, Mozilla Data Collective) | CC0 (qidiruv natijasidagi ma'lumotnoma matniga ko'ra: 230 040 klip, 265.75 soat, 101.01 soat tasdiqlangan, 2 299 ovoz egasi). Shartlar: ma'lumotni qayta joylamaslik va ovoz egalarini aniqlashga urinmaslik | S: sahifa JavaScript bilan yuklanadi, to'liq o'qilmadi; 23.0 versiyasi ro'yxatida ham CC0-1.0. Yuklash uchun Data Collective hisobi va shartlarni qabul qilish kerak bo'lishi mumkin | CC0 bo'lgani uchun asosan ha; shartlar o'qilmaguncha tasdiqlanmagan | candidate (egasi sahifani o'qib, shartni qabul qilgach) |
| MIT Environmental Impulse Responses (HF `davidscripka/...`, asli MIT IR Survey) | HF: "unknown"; asl sahifada litsenziya yo'q | P: ikkala sahifa | noma'lum | **stopped, not used** |
| BIRD (Big Impulse Response Dataset) | tekshirilmadi | yo'q | noma'lum | **stopped, not used** |
| FSD50K | upstream hujjatiga ko'ra "turli CC litsenziyalari" (klipga qarab, NC bo'lishi mumkin) | S: upstream `documentation/data_sources.md` | klip bo'yicha | **stopped, not used** |
| FMA | upstream: CC BY 4.0; trek litsenziyalari alohida bo'lishi mumkin | S | tekshirilmagan | **stopped, not used** |
| WHAM! | CC BY-NC 4.0 (noncommercial) | S: upstream hujjati | yo'q (NC) | **stopped, not used** |
| VOiCES, DiPCo, CHiME6, LibriSpeech | upstream: VOiCES CC BY 4.0, DiPCo CDLA-Permissive-1.0 | S: upstream hujjati | tekshirilmagan | **stopped, not used** |
| `kahrendt/microwakeword` (HF, oldindan hisoblangan salbiy xususiyatlar) | CC BY-NC 4.0 | P: HF sahifasi | NC: yo'q | **stopped, not used** |
| AudioSet | audio taqsimlanmaydi (YouTube) | S | n/a | not used |

Eslatma: upstream notebook'i o'zi yuklaydigan ma'lumotlar haqida "turli litsenziyalar va cheklovlar aralashmasi; shu bilan o'qitilgan model faqat notijorat shaxsiy foydalanish uchun" deb ogohlantiradi. Shu sabab upstream notebook'ining ma'lumot yuklash qadamlari bu repoda ishlatilmaydi.

## Bizning ma'lumotlar

| Nom | Litsenziya | Holat |
| --- | --- | --- |
| Rozilik bilan yozilgan klip'lar (`source: recorded`) | `LicenseRef-Moby-Consented-Recording`: ichki, rozilik matni (versiya `2026-11-v1`) bilan boshqariladi. Klip'lar nashr qilinmaydi; model nashr qilinishi mumkin | allowed (faqat rozilik bilan) |
| `mobyww.data.sample` sinus tonlari | CC0-1.0 (kod bilan hosil qilinadi, ovoz yo'q) | allowed |
| W0 smoke-run sintetik ohanglari | ularni `scripts/upstream_smoke.py` hosil qiladi, saqlanmaydi, CC0 | allowed |

## Robot tomoni (eslatma)

ESP-SR / WakeNet ishlatilmaydi. Espressif sahifasi litsenziyani ko'rsatmadi; bundan tashqari ular xitoy, ingliz, yapon, frantsuz TTS ro'yxatida o'zbek tilini e'lon qilmagan (reja). Holat: stopped, not used.

## Qoida

Yangi to'plam, dvigatel, RIR yoki kutubxona: avval shu jadvalga birlamchi manba bilan yoziladi, "candidate" yoki "allowed" bo'lmasa `data/` ga olinmaydi. Manifest validatori litsenziyasiz qatorni rad etadi.
