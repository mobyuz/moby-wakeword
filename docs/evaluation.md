# Baholash

Holat: **qoralama**. Hech narsa o'lchanmagan: hech qanday model yo'q, shuning uchun aniqlik, recall va yolg'on uyg'onish haqida raqam yo'q. Quyida W4 da amalga oshiriladigan reja va hozirgi tasdiqlanmagan chegaralar.

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
- Sintetik (TTS) bilan o'qitilgan model sintetik val/test bilan o'lchansa, natija optimistik chiqadi (upstream notebook ham shuni ogohlantiradi). Baholash uchun haqiqiy yozuv yoki boshqa TTS dvigateli kerak.
- Salbiy soat to'plami hali yo'q: [licenses.md](licenses.md) bo'yicha ruxsat etilgan manba topilmagan.

## Chegaralar (`configs/targets.yaml`) , TASDIQLANMAGAN

Reja boshlang'ich takliflari; egasi tasdiqlamaguncha majburiy emas.

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
