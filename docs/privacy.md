# Maxfiylik

Holat: W1 qoralamasi. Huquqiy tekshiruvdan o'tmagan; egasi tasdiqlashi kerak.

## Nima yig'iladi va nima uchun

- Qisqa audio klip'lar (ovoz egasi aytgan ibora yoki erkin nutq) wake word modelini o'qitish va baholash uchun.
- Har klip uchun metadata: taxallus, masofa, ovoz balandligi, xona, mikrofon nomi, sana, til.
- Rozilik yozuvi: taxallus, rozilik identifikatori, matn versiyasi, ruxsat doirasi, sana, qaytarib olingan bo'lsa sanasi. Ixtiyoriy `identity_note` faqat lokal, erkin matn; shaxs aniqlanadigan yagona joy. Uni to'ldirmaslik mumkin.

Yig'ilmaydi: haqiqiy ism klip va manifestlarda, joylashuv, hisob ma'lumotlari. Klip'lar hech qaerga yuklanmaydi.

## Qayerda saqlanadi

Faqat `data/` ichida, loyiha egasining mashinasida: `data/recordings/`, `data/consent/`, `data/manifests/`. `.gitignore` bularni yopadi, `git add -f` bilan chetlab o'tishni esa pre-commit tekshiruvi (`.githooks/pre-commit`, `mobyww.guard`) va CI (`--tracked`) to'xtatadi. Tekshiruv audio fayllarni, og'irliklarni va xususiyat papkalarini, rozilik va shaxs fayllarini, litsenziyasi noma'lum sinov audiosini bloklaydi; testlar bu bloklashni haqiqiy git bilan isbotlaydi. Cheklov: tekshiruv fayl yo'li va kengaytmasiga qaraydi, ichidagi mazmunga emas. Fayl nomini o'zgartirib yuborish uni aldashi mumkin, shuning uchun sharh (review) kerak.

Kelajakda diskdagi zaxira nusxa bo'lsa, u shifrlangan bo'lishi va shu hujjatda yozilishi kerak. Hozir zaxira yo'q.

## Nima nashr qilinadi

Faqat model (og'irliklar), uning manifesti, xeshlar va baholash hisoboti. Klip'lar va rozilik fayllari hech qachon. Rozilik matni modelning ochiq litsenziya bilan nashr etilishi mumkinligini aytadi. Model o'qitilgan nutqni to'liq qayta tiklay olishi haqida da'vo qilinmaydi ham, inkor qilinmaydi: bu o'lchanmagan.

## Ovoz egasi o'chirishni qanday so'raydi

1. Loyiha egasiga taxallusini ayting (`spk-xxxxxx`).
2. Egasi `mobyww-record withdraw --speaker <taxallus>` ni ishga tushiradi: rozilik "qaytarib olingan" deb belgilanadi, validator bu ovoz egasining qatorlarini rad etadi, yozish vositasi yangi klip yozmaydi.
3. Egasi `data/recordings/<taxallus>/` papkasini va manifestdagi qatorlarni o'chiradi, rozilik yozuvini saqlab qoladi yoki o'chiradi (so'rov bo'yicha).
4. Shu ovoz egasi bilan o'qitilgan model qayta o'qitilmaguncha o'zgarmaydi; keyingi reliz uni chiqarib tashlab o'qitiladi. Allaqachon nashr etilgan model "o'qitilmagan" bo'lib qolmaydi: rozilik matni buni ochiq aytadi.

Muddat va aloqa kanali hali belgilanmagan (egasi qaror qiladi).

## Ochiq qoldirilgan masalalar

- Voyaga yetmaganlar yozuvi: qoida yo'q; hozircha faqat kattalar.
- Ma'lumot saqlash muddati: belgilanmagan.
- Rozilik matni yuridik ko'rikdan o'tmagan.
