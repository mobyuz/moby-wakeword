# Robot bilan shartnoma (firmware)

Holat: **qoralama**. Bu hujjat nima aytilishi kerakligini belgilaydi; o'lchangan yoki robotda sinalgan narsa yo'q. "OCHIQ" bilan belgilangan joylar hal etilmagan va to'qib chiqarilmagan.

## Reliz nimani beradi (reja, W5/W6 da yoziladi)

Kvantlangan oqimli TFLite modeli; manifest JSON (upstream joriy formatiga mos: ibora, o'qitilgan tillar, ehtimollik chegarasi, sirpanuvchi oyna, xususiyat qadami, tensor arena hajmi, namuna chastotasi, versiya, o'qitish commit'i, ma'lumot manifesti xeshi, baholash xulosasi); SHA-256 xeshlar; baholash hisoboti. Bularning birortasi hali mavjud emas.

## Firmware nima qilishi kerak

1. **Kirish.** 16 kHz, mono, 16 bit. Upstream README'ga ko'ra xususiyatlar 16 kHz xom audiodan, 30 ms oynada, 10 ms qadam bilan 40 ta xususiyat (TFLite Micro micro_speech preprocessor), oqimli model har 30 ms da xulosa chiqaradi (konvolyutsiya uchta 10 ms bo'lak ustidan yuradi). Aniq qiymatlar eksport qilingan manifestdan olinadi, bu yerdan emas.
2. **24 bitdan 16 bitga o'tkazish. OCHIQ.** INMP441 24 bitli qiymatni 32 bitli slotda beradi. Robotdagi o'tkazish (siljish va masshtab) `moby-firmware` repozitoriysida yashaydi (`main/moby_pcm.h`, `main/moby_pcm.c`) va F4 bosqichida yakunlanmoqda. Augmentatsiyadagi qurilma yo'li taqlidi (W2) aynan shu o'tkazishni ishlatishi shart, shuning uchun siljish va masshtab qiymati bu yerga **ko'chirilmaydi**: F4 yakunlangach `configs/augmentation.yaml` dagi `TBD-F4` ga bitta manbadan qo'yiladi, va firmware testi bilan bu repo orasida bir xil kirish-chiqish juftliklari tekshiriladi.
3. **Chegara va sirpanuvchi oyna. OCHIQ.** Qiymatlar baholashdan (W4) keyin tanlanadi va manifestga yoziladi. Firmware ularni manifestdan o'qishi kerak, qotirib qo'ymasligi kerak. Qoida (upstream): model ketma-ket bir necha oynada ehtimollikni chegaradan yuqori bersa, ibora aytilgan hisoblanadi.
4. **Xotira. OCHIQ.** Model PSRAM'da va alohida bo'limdan yuklanadi (reja); tensor arena hajmi manifestdan. Bo'lim o'lchami va yuklash mexanizmi firmware tomonida hal etilmagan. Upstream ESPHome komponenti PSRAM'li ESP32-S3 talab qiladi (reja manbasi).
5. **O'z ovozi va mute.** Robot gapirayotganda detektsiya o'chiriladi yoki natija e'tiborsiz qoldiriladi. Apparat mute hammasidan ustun: mute holatida detektsiya natijasi hech narsani ishga tushirmaydi va ovoz hech qaerga yuborilmaydi. Bu firmware'ning mavjud maxfiylik invariantlari bilan mos bo'lishi kerak; shartnomada ularni takrorlamaymiz.
6. **Uyg'onish natijasi.** Uyg'onishda firmware nima qilishi (push-to-talk bilan bir xil seans ochishmi) protokol va firmware repolarida hal etiladi; bu yerda belgilanmagan.

## Host replay vositasi (W5, hali yo'q)

`mobyww.replay` eksport qilingan modelni WAV fayl ustida qurilmadagi kabi (16 kHz, xuddi shu qadam) ishga tushirib, vaqt belgilari bilan aniqlashlarni chiqaradi. Oltin test: bir necha qisqa WAV (faqat o'zimizniki yoki sintetik; `tests/golden/PROVENANCE.yaml` da litsenziyasi bilan) va kutilgan vaqt belgilari CI'da tekshiriladi. Firmware muhandislari qurilma natijasini shu bilan solishtiradi. Hozir kod yo'q.

## Ochiq masalalar ro'yxati

| # | Masala | Egasi | Qachon |
| --- | --- | --- | --- |
| 1 | 24 -> 16 bit siljish va masshtab qiymati | moby-firmware (F4) | F4 yakuni, W2 dan oldin |
| 2 | Chegara va sirpanuvchi oyna | W4 baholash | W4 |
| 3 | Model bo'limi o'lchami va yuklash | moby-firmware | W5 dan oldin |
| 4 | Uyg'onish natijasi bo'yicha xatti-harakat | firmware + protokol | W5 dan oldin |
| 5 | Manifest JSON'ning aniq maydonlari (upstream joriy formati bilan tekshirilishi kerak) | W5 | W5 |
