# Baholash

Holat: **qoralama**. Hech narsa o'lchanmagan: hech qanday model yo'q, shuning uchun aniqlik, recall va yolg'on uyg'onish haqida raqam yo'q. Quyida W4 da amalga oshiriladigan reja va egasi tasdiqlagan chegaralar.

## Ko'rsatkichlar

Har natija o'zining test hajmi (klip soni, ovoz egalari soni, soat) bilan beriladi; hajm ko'rsatilmagan natija hisobotga kirmaydi.

| Ko'rsatkich | Ma'nosi |
| --- | --- |
| Soatiga yolg'on uyg'onish | kamida 10 soat (sozlanadi) o'qitishda ishlatilmagan salbiy audioda; nutq, televizor/podkast, musiqa, shovqin bo'yicha alohida |
| Recall | umumiy, masofa (30 sm, 1 m, 3 m), shovqin darajasi va ovoz egasi bo'yicha |
| Aniqlash kechikishi | ibora tugagandan detektsiyagacha vaqt |
| Ishlash nuqtasi | ehtimollik chegarasi va sirpanuvchi oyna hajmi bo'yicha egri chiziq va tavsiya etilgan nuqta |
| Qat'iy salbiylar | "mobil", "mobile", "mobi", "moby dick", "hey mom", "hey bobby", "hey toby" necha marta ishga tushadi |
| Model resursi | hajm (KB), tensor arena hajmi, operatsiyalar soni, TFLite Micro qo'llab-quvvatlamaydigan operatsiyalar |
| CPU yuklamasi, quvvat | **faqat haqiqiy platada o'lchanadi**; o'lchanmaguncha hisobotda belgilangan bo'sh joy ("unmeasured") turadi |

Egasi 2026-11-05 da `hey moby` ni tanladi, shuning uchun faqat shu ibora baholanadi; boshqa nomzodlar taqqoslanmaydi.

## To'plamlar

- Bo'lish ovoz egalari bo'yicha: train, val, test (bitta odam ikki to'plamda emas, validator tekshiradi).
- Test to'plamiga faqat yakuniy hisobotda tegiladi; sozlash val'da.
- Stress to'plami: shovqin, uzoq masofa, qat'iy salbiylar.
- **Test to'plami: egasining haqiqiy yozuvlari** (~20 daqiqa, lokal; Colab'ga yoki boshqa joyga hech qachon yuklanmaydi). Sintetik (TTS) bilan o'qitilgan model sintetik val/test bilan o'lchansa, natija optimistik chiqadi (upstream notebook ham shuni ogohlantiradi): o'qitish va baholash bir xil ovozlar oilasidan. Shu sabab `mobyww.synth` manifest yozuvchisi sintetik klip'lar uchun test bo'limini sukut bo'yicha 0% qiladi (faqat train va val); sintetik val faqat o'qitishni kuzatish uchun, hisobotdagi raqam emas. Hisobotga faqat haqiqiy yozuvlardagi natija kiradi, hajmi (klip, ovoz egalari soni) bilan.
- Haqiqiy yozuvlar ~20 daqiqa va bir necha ovoz egasi bo'lgani uchun ishonch oralig'i keng bo'ladi; bu hisobotda aytiladi.
- Salbiy soat to'plami hali yo'q. MUSAN ([licenses.md](licenses.md), CC BY 4.0, allowed) nutq qismi (maqolaga ko'ra ~60 soat) nomzod, lekin kamida 10 soat o'qitishda ishlatilmasligi va alohida ajratilishi kerak; bu hali qilinmagan. Colab daftari MUSAN'dan faqat qisqa bo'laklarni (fayl boshiga cheklangan) o'qitish salbiylari sifatida oladi.

## Chegaralar (`configs/targets.yaml`), egasi tasdiqlagan

Egasi bu qiymatlarni tasdiqlagan (`status: confirmed-by-owner`). Ular reliz darvozasi uchun majburiy; hozircha ularni tekshiradigan kod yo'q va hech bir model ularni o'tmagan, chunki model yo'q.

| Chegara | Taklif |
| --- | --- |
| Yolg'on uyg'onish | soatiga 1 tadan ko'p emas |
| Jim xonada 1 m | recall kamida 90% |
| 3 m yoki o'rtacha shovqin | recall kamida 75% |

Reliz chegaralar bajarilmasa to'xtaydi; faqat aniq override bayrog'i (`--override-gates`) bilan va hisobotda qayd etilgan holda o'tadi. Bu mexanizm W4 da yoziladi; hozir kod yo'q.

## Upstream o'zi nima o'lchaydi

microWakeWord o'qitish paytida ambient to'plamlarda yolg'on qabullarni soatiga taxmin qiladi (uzun klip'larni 100 ms qadam bilan bo'lib). Upstream README'ga ko'ra bu streaming modelning real soatiga ko'rsatkichining mukammal bahosi emas. Bizning baholash vositasi (W4) streaming modelni to'g'ridan-to'g'ri uzun audio ustida ishga tushiradi.

## W0 smoke-run haqida

W0 da o'qitilgan bitta kichik model sintetik sinus-ohang naqshi ustida bo'lgan; u nutq modeli emas va undan hech qanday ko'rsatkich keltirilmaydi.
