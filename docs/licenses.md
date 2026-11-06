# Litsenziyalar

Tekshirilgan sana: 2026-11-05; W2 qismi (TTS, MUSAN, ReverbDB, Common Voice, o'zbek TTS) 2026-11-06. Holat ustuni: **allowed** (ishlatilishi mumkin), **candidate** (yo'l bor, lekin yuklab olishdan oldin birlamchi manbada tasdiqlanishi kerak), **stopped, not used** (aniq emas yoki cheklov bor; ishlatilmaydi). "Tarqatish" ustuni: o'qitilgan modelni (GPL-3.0-only repodan GitHub Release orqali) tarqatish mumkinmi.

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

## TTS dvigatellari (W2; faqat Colab'da ishlatiladi, mahalliy mashinada generatsiya yo'q)

2026-11-06 da birlamchi manbalardan o'qildi: GitHub API va xom fayllar, Hugging Face API (`/api/models/...`, `/api/datasets/...`) va xom `MODEL_CARD`/`README.md`, PyPI JSON. O'qilmagan qismlar har qatorda yozilgan. Hech bir model yuklab olinmadi va hech narsa generatsiya qilinmadi.

| Nom | Litsenziya | Manba | Tarqatish | Holat |
| --- | --- | --- | --- | --- |
| `rhasspy/piper-sample-generator` (kod), commit `2971426a55072f7d22fec416ca7800df8bd23207` (3.2.0) | MIT (`LICENSE.md`; GitHub API `spdx_id: MIT`) | P: https://github.com/rhasspy/piper-sample-generator | kodga taalluqli emas | allowed |
| `piper-tts` 1.3.0 (PyPI; espeak-ng ni o'z ichiga oladi) | GPL-3.0-or-later (PyPI metadata'si); GitHub API: `OHF-Voice/piper1-gpl` va `espeak-ng/espeak-ng` GPL-3.0 | P: https://pypi.org/pypi/piper-tts/1.3.0/json | Faqat Colab'da ishlaydi, repoga kod qo'shilmaydi | allowed (vosita sifatida) |
| `en_US-libritts_r-medium.onnx` (904 ovozli Piper ovozi; `rhasspy/piper-voices` revision `e7229f7d33a10e145e109b886d661984201ac3c4`; 78 580 914 bayt; SHA-256 `10bb85e071d616fcf4071f369f1799d0491492ab3c5d552ec19fb548fac13195`, HF LFS metadata'sidan) | repo metadata'si `license: mit`; shu ovozning `MODEL_CARD` fayli: ma'lumot LibriTTS-R (OpenSLR 141), CC BY 4.0; "Fine-tuned from English lessac medium on train-clean-360" | P: https://huggingface.co/rhasspy/piper-voices/raw/main/en/en_US/libritts_r/medium/MODEL_CARD | Atribusiya kerak ([ATTRIBUTION.md](ATTRIBUTION.md)). Kartada chiqish audiosiga cheklov yo'q. O'qilmagan: asos "lessac" ovozining ma'lumot litsenziyasi | allowed (atribusiya bilan) |
| `en_US-libritts_r-medium.pt` (piper-sample-generator v2.0.0 reliz fayli, 204 089 915 bayt) | Reliz sahifasida litsenziya **ko'rsatilmagan**. HF `rhasspy/piper-checkpoints` (u *dataset* repo, `license: mit`; MODEL_CARD: LibriTTS-R CC BY 4.0, "Trained from scratch", "Requires Piper 1.5") boshqa, yangi tekshiruv nuqtasi; reliz `.pt` aynan shundan olinganini sahifa isbotlamaydi (README faqat "exported from checkpoint" deydi) | P: https://github.com/rhasspy/piper-sample-generator/releases/tag/v2.0.0 va https://huggingface.co/datasets/rhasspy/piper-checkpoints/raw/main/en/en_US/libritts_r/medium/MODEL_CARD. Eslatma: `https://huggingface.co/api/models/rhasspy/piper-checkpoints` 401 beradi, chunki bu model emas, dataset repo; to'g'ri API `/api/datasets/rhasspy/piper-checkpoints` (200) | noma'lum | **stopped, not used** (o'rniga yuqoridagi `.onnx`) |
| Piper rus ovozlari `ru_RU-denis-medium`, `ru_RU-dmitri-medium` (`rhasspy/piper-voices`, SHA-256 `15fab56e11a097858ee115545d0f697fc2a316c41a291a5362349fb870411b0a` va `f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb`) | `MODEL_CARD`: ma'lumot `OHF-Voice/voice-datasets`, CC0; shu repo README: "These datasets are licensed under CC0 (public domain)" | P: https://huggingface.co/rhasspy/piper-voices/raw/main/ru/ru_RU/denis/medium/MODEL_CARD, `.../dmitri/medium/MODEL_CARD`, https://github.com/OHF-Voice/voice-datasets | CC0: atribusiya shart emas | allowed |
| Piper `ru_RU-irina-medium` | `MODEL_CARD`: "License: Unknown" (RHVoice) | P: `.../ru/ru_RU/irina/medium/MODEL_CARD` | noma'lum | **stopped, not used** |
| Piper `ru_RU-ruslan-medium` | `MODEL_CARD`: CC BY-NC-SA 4.0 (noncommercial) | P: `.../ru/ru_RU/ruslan/medium/MODEL_CARD` | NC: yo'q | **stopped, not used** |
| Piper o'zbek ovozi | `rhasspy/piper-voices` til ro'yxatida `uz` yo'q (API `tags`) | P: https://huggingface.co/api/models/rhasspy/piper-voices | n/a | mavjud emas |
| Navoiy TTS (`aisha-org/navoiy-tts`, revision `9445f41ee5f38bd3a34eb16b0279dc05f9997fc2`; faqat LLM nuqtasi `emotion_600h_joint.pt`, 1.88 GiB; mustaqil emas, CosyVoice2 ish vaqti va asos model kerak) | Apache-2.0 (karta, `LICENSE` standart Apache-2.0 matni, `NOTICE`: CosyVoice va CosyVoice2-0.5B dan olingan, Apache-2.0). Asos `FunAudioLLM/CosyVoice2-0.5B`: HF `license: apache-2.0`; kod `FunAudioLLM/CosyVoice`: GitHub API Apache-2.0. Foydalanish qoidalari (loyiha qo'llanmasi): ovozni klonlashdan oldin ovoz egasining aniq roziligi; qalbakilashtirish, aldov, firibgarlik va ovozli autentifikatsiyani aylanib o'tish taqiqlanadi; sintetik ekani aytiladi | P: https://huggingface.co/aisha-org/navoiy-tts (API, README, NOTICE, LICENSE), https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B, https://voicelab.uz/blog/navoiy-tts-open-source-uzbek-text-to-speech. **O'qilmadi:** o'qitish ma'lumotlari. Karta ko'rsatgan `aisha-org/uzbek-tts-corpus-v1` va `aisha-org/uzbek-emotion-tts` API'si 401 beradi; CosyVoice2 kartasida o'qitish ma'lumoti litsenziyasi yo'q (CosyVoice README "Disclaimer: academic purposes" demo kontentga oid) | model/kod uchun Apache-2.0 tarqatishga yo'l qo'yadi; o'qitish ma'lumotlari noma'lum | candidate (yuklanmagan; ma'lumot manbasi aniqlanmaguncha va egasi qaror qilmaguncha ishlatilmaydi) |
| Chatterbox va Chatterbox Multilingual (`ResembleAI/chatterbox`, revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`) | MIT (karta va API). Karta: "Trained on 0.5M hours of cleaned data", "Prompts are sourced from freely available data on the internet"; har chiqish PerTh suv belgisi bilan. Til ro'yxati 23 ta, o'zbek yo'q (rus bor) | P: https://huggingface.co/ResembleAI/chatterbox (API, README) | MIT tarqatishga yo'l qo'yadi; o'qitish ma'lumotlari litsenziyasi aytilmagan | candidate (yuklanmagan) |
| Chatterbox-MTL Central-Asian (`Tohirju/chatterbox-mtl-central-asian`, revision `e31153f333783661c159525a1507faf49da0d058`; tg, uz, kk, ky) | API metadata'si `license: mit`, lekin repo qo'lda yopilgan (`gated: manual`; forma https://saidzoda.com/access). `README.md` 401: karta, o'qitish ma'lumotlari va qo'shimcha shartlar o'qilmadi | P: https://huggingface.co/api/models/Tohirju/chatterbox-mtl-central-asian (faqat metadata) | noma'lum | **stopped, not used** |
| Boshqa o'zbek Chatterbox fine-tune'lari (`Tohirju/chatterbox-mtl-uzbek`, `UAzimov/Uzbek-tts-chatterbox`, `abbos2332/uzbek-tts`) | birlamchi manbada tekshirilmadi | S: veb-qidiruv natijasi | noma'lum | **stopped, not used** |
| LibriTTS-R (OpenSLR 141) ma'lumot to'plamining o'zi | CC BY 4.0 (sahifa: "License: CC BY 4.0") | P: https://www.openslr.org/141/ | Piper ovozini o'qitgan ma'lumot; to'plam yuklanmaydi (dev_clean 1.3 GB dan train_other_500 46 GB gacha), faqat atribusiya | not used directly |

### Generatsiya qilingan audio GPL-3.0-only model uchun ishlatilishi mumkinmi

Yuridik maslahat emas; o'qilgan shartlarga ko'ra:

- Dastur yoki model litsenziyasi (MIT, Apache-2.0, GPL-3.0 `piper-tts`) o'zi chiqargan audioga o'z-o'zidan tarqalmaydi deb o'qiymiz. Apache-2.0 va MIT matnlarida chiqishga cheklov yo'q; o'qilgan model kartalarida ham yo'q (Navoiy: faqat ovozni klonlash roziligi va suiiste'mol taqiqi; biz klonlamaymiz).
- Repo va model GPL-3.0-only. TTS ovozlari va audio repoga qo'shilmaydi va tarqatilmaydi, faqat Colab'da ishlatiladi; shuning uchun GPL bilan to'qnashuv yo'q. Nashr qilinadigan narsa faqat o'qitilgan model.
- Qolgan xavf: TTS modellari o'qitilgan ma'lumotlar litsenziyasi. Piper `libritts_r` (CC BY 4.0) va rus `denis`/`dmitri` (CC0) uchun o'qildi. Navoiy/CosyVoice2 va Chatterbox uchun o'qitish ma'lumotlari ochiq emas yoki o'qilmadi; shuning uchun ular "candidate" va hozir ishlatilmaydi.
- Natijada ruxsat etilgan sintetik ijobiy manba: Piper `en_US-libritts_r-medium`, `ru_RU-denis-medium`, `ru_RU-dmitri-medium`. O'zbek aksenti hozircha shu ovozlarda `hei mobi` kabi imlo variantlari bilan taqlid qilinadi; haqiqiy o'zbek TTS yo'q. Bu cheklov hisobotlarda aytilishi shart.

## Salbiy va fon ma'lumotlari

| Nom | Litsenziya | Manba | Tarqatish | Holat |
| --- | --- | --- | --- | --- |
| Mozilla Common Voice, o'zbek (Common Voice Scripted Speech 25.0 - Uzbek, Mozilla Data Collective) | CC0 (qidiruv natijasidagi ma'lumotnoma matniga ko'ra: 230 040 klip, 265.75 soat, 101.01 soat tasdiqlangan, 2 299 ovoz egasi). Shartlar: ma'lumotni qayta joylamaslik va ovoz egalarini aniqlashga urinmaslik | S: sahifa (https://mozilladatacollective.com/datasets/cmn29fa4y0150o1073fqww73p) JavaScript bilan yuklanadi, 2026-11-06 da matni olinmadi; 23.0 versiyasi ro'yxatida ham CC0-1.0. P (2026-11-06): `GET https://mozilladatacollective.com/api/datasets/cmn29fa4y0150o1073fqww73p` -> 403 `Access to this dataset has not been granted`, ya'ni yuklash uchun hisob va shartni qabul qilish kerak. Shartlar matni o'qilmadi | CC0 bo'lgani uchun asosan ha; shartlar o'qilmaguncha tasdiqlanmagan | candidate (egasi sahifani o'qib, shartni qabul qilgach; [qadamlar](data-collection.md#common-voice-ozbek-egasi-qolda)). Skript uni yuklamaydi |
| MUSAN (OpenSLR 17): musiqa, nutq, shovqin; `musan.tar.gz` 11 086 114 085 bayt, MD5 `0c472d4fc0c5141eca47ad1ffeb2a7df` | CC BY 4.0 (OpenSLR sahifasi va `info.txt`). Maqola (arXiv 1510.08484): hamma fayllar Creative Commons yoki AQSh public domain, tijoriy foydalanishga ruxsat berilganlari; Librivox nutq PD, Free Sound shovqin PD, SoundBible CC, musiqa Jamendo/FMA va boshqalar turli CC; har kichik papkada `LICENSE` fayli fayl bo'yicha atribusiyani beradi | P: https://www.openslr.org/17/, https://www.openslr.org/resources/17/info.txt, https://www.openslr.org/resources/17/checksum.md5, https://arxiv.org/abs/1510.08484 (PDF o'qildi). **O'qilmadi:** tarball ichidagi fayl bo'yicha `LICENSE`lar (11 GB yuklanmadi) | CC BY 4.0: atribusiya ([ATTRIBUTION.md](ATTRIBUTION.md)); ichki fayl bo'yicha litsenziyalar birinchi yuklashda arxivlanadi | allowed (korpus darajasida, atribusiya bilan). Fayl bo'yicha litsenziyalar tekshirilmagan; Colab skripti `LICENSE` fayllarini natija yoniga nusxalaydi |
| BUT Speech@FIT Reverb Database, `BUT_ReverbDB_rel_19_06_RIR-Only.tgz` (RIR, xona shovqini, metadata; 9 xona, 31 mikrofon; 16 kHz mono 16 bit WAV); 9 308 593 693 bayt (sahifa "8.7 GB") | CC BY 4.0 (loyiha sahifasi: "The database has CC-BY 4.0 license"). `read_me.txt` sarlavhasi Apache-2.0 deydi (skriptlar/metafayllar uchun ko'rinadi, ma'lumot litsenziyasi sahifadagi CC BY 4.0 deb o'qildi). Sahifada nashr qilingan xesh yo'q | P: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database, https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz (HEAD: 200, `Accept-Ranges: bytes`, hajm), https://merlin.fit.vutbr.cz/ReverbDB/read_me.txt | CC BY 4.0: atribusiya va maqolani keltirish ([ATTRIBUTION.md](ATTRIBUTION.md)) | allowed (atribusiya bilan). Xesh birinchi yuklashda hisoblanib qayd etiladi; "LibriSpeech-Only" (117 GB) va HUB5 (229 GB) ishlatilmaydi |
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
