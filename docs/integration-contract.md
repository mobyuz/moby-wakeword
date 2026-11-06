# Robot bilan shartnoma (firmware)

Holat: **qoralama**. Bu hujjat nima aytilishi kerakligini belgilaydi; o'lchangan yoki robotda sinalgan narsa yo'q. "OCHIQ" bilan belgilangan joylar hal etilmagan va to'qib chiqarilmagan.

## Reliz nimani beradi (reja, W5/W6 da yoziladi)

Kvantlangan oqimli TFLite modeli; manifest JSON (upstream joriy formatiga mos: ibora, o'qitilgan tillar, ehtimollik chegarasi, sirpanuvchi oyna, xususiyat qadami, tensor arena hajmi, namuna chastotasi, versiya, o'qitish commit'i, ma'lumot manifesti xeshi, baholash xulosasi); SHA-256 xeshlar; baholash hisoboti. Bularning birortasi hali mavjud emas.

## Firmware nima qilishi kerak

1. **Kirish.** 16 kHz, mono, 16 bit. Upstream README'ga ko'ra xususiyatlar 16 kHz xom audiodan, 30 ms oynada, 10 ms qadam bilan 40 ta xususiyat (TFLite Micro micro_speech preprocessor), oqimli model har 30 ms da xulosa chiqaradi (konvolyutsiya uchta 10 ms bo'lak ustidan yuradi). Aniq qiymatlar eksport qilingan manifestdan olinadi, bu yerdan emas.
2. **24 bitdan 16 bitga o'tkazish.** INMP441 24 bitli qiymatni 32 bitli slotda beradi. Asl manba `moby-firmware` dagi `main/moby_pcm.h` va `main/moby_pcm.c` (`claude/f4-ptt-audio` branchi, hali merge qilinmagan, platada sinalmagan). Qiymatlar `configs/augmentation.yaml` da: faqat chap slot, 32 bitli slot arifmetik ravishda 16 bitga o'ngga suriladi (yaxlitlash va dither yo'q), kuchaytirish 256/256 va to'yinish bilan. DC olib tashlanmaydi, filtr yo'q. Mikrofon ishga tushgach birinchi 260 ms tashlab yuboriladi. Augmentatsiyadagi qurilma yo'li aynan shuni ishlatadi: `mobyww.augment.device` (`to_slot`, `slot_to_pcm16`, `convert_stereo`, `drop_warmup`) `moby_pcm.c` dagi `moby_pcm_sample` ni bit-bit takrorlaydi (arifmetik surish = pol bo'yicha bo'lish, salbiy qiymatlar uchun ham; kuchaytirish 256 dan boshqa bo'lsa pol bo'yicha 256 ga bo'lish va to'yinish; 4096 dan katta kuchaytirish 4096 ga qisqaradi). Isbot: `tests/test_augment_device.py` da qo'lda `moby_pcm.c` dan chiqarilgan kirish-chiqish juftliklari (salbiy qiymatlar, to'yinish, pol), C kodining aynan Python nusxasi bilan 20 000 tasodifiy slot, barcha 65 536 yuqori-16 qiymat uchun xesh (konversiya o'zgarsa test yiqiladi) va, `moby-firmware` repo shu repo yonida turgan bo'lsa, `moby_pcm.h` dagi konstantalar bilan solishtirish. **Hali yo'q:** firmware tomonida shu juftliklar bilan test (u repo bu yerda o'zgartirilmaydi) va platada tekshiruv. Eslatma: augmentatsiyadagi ixtiyoriy 7200 Hz past chastota filtri firmware'ga tegishli emas; u mikrofonning o'lchanmagan taxminiy modeli ([augmentation.yaml](../configs/augmentation.yaml)), firmware filtr qo'llamaydi.
3. **Chegara va sirpanuvchi oyna. OCHIQ.** Qiymatlar baholashdan (W4) keyin tanlanadi va manifestga yoziladi. Firmware ularni manifestdan o'qishi kerak, qotirib qo'ymasligi kerak. Qoida (upstream): model ketma-ket bir necha oynada ehtimollikni chegaradan yuqori bersa, ibora aytilgan hisoblanadi.
4. **Xotira. OCHIQ.** Model PSRAM'da va alohida bo'limdan yuklanadi (reja); tensor arena hajmi manifestdan. Bo'lim o'lchami va yuklash mexanizmi firmware tomonida hal etilmagan. Upstream ESPHome komponenti PSRAM'li ESP32-S3 talab qiladi (reja manbasi).
5. **O'z ovozi va mute.** Robot gapirayotganda detektsiya o'chiriladi yoki natija e'tiborsiz qoldiriladi. Apparat mute hammasidan ustun: mute holatida detektsiya natijasi hech narsani ishga tushirmaydi va ovoz hech qaerga yuborilmaydi. Bu firmware'ning mavjud maxfiylik invariantlari bilan mos bo'lishi kerak; shartnomada ularni takrorlamaymiz.
6. **Uyg'onish natijasi.** Uyg'onishda firmware nima qilishi (push-to-talk bilan bir xil seans ochishmi) protokol va firmware repolarida hal etiladi; bu yerda belgilanmagan.

## Host replay vositasi (W5, hali yo'q)

`mobyww.replay` eksport qilingan modelni WAV fayl ustida qurilmadagi kabi (16 kHz, xuddi shu qadam) ishga tushirib, vaqt belgilari bilan aniqlashlarni chiqaradi. Oltin test: bir necha qisqa WAV (faqat o'zimizniki yoki sintetik; `tests/golden/PROVENANCE.yaml` da litsenziyasi bilan) va kutilgan vaqt belgilari CI'da tekshiriladi. Firmware muhandislari qurilma natijasini shu bilan solishtiradi. Hozir kod yo'q.

## Ochiq masalalar ro'yxati

| # | Masala | Egasi | Qachon |
| --- | --- | --- | --- |
| 1 | 24 -> 16 bit o'tkazish uchun bir xil juftliklar testi. Bu repo tomoni bajarildi (`tests/test_augment_device.py`); firmware tomoni ochiq | firmware | W5 dan oldin |
| 2 | Chegara va sirpanuvchi oyna | W4 baholash | W4 |
| 3 | Model bo'limi o'lchami va yuklash | moby-firmware | W5 dan oldin |
| 4 | Uyg'onish natijasi bo'yicha xatti-harakat | firmware + protokol | W5 dan oldin |
| 5 | Manifest JSON'ning aniq maydonlari (upstream joriy formati bilan tekshirilishi kerak) | W5 | W5 |
