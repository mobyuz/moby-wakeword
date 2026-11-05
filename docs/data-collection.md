# Ma'lumot yig'ish

Holat: W1 qoralamasi. Bu repoda haqiqiy audio yozilmagan; vosita faqat testlarda soxta audio manba bilan ishlatilgan.

## Nima yig'iladi

| Tur | Manba | Yorliq |
| --- | --- | --- |
| Iborani yozdirish (30 sm, 1 m, 3 m; sekin, normal, baland ovoz) | `mobyww-record session` | `wake` |
| Qat'iy salbiylar ("mobil", "mobile", "mobi", "moby dick", "hey mom", "hey bobby", "hey toby") | `mobyww-record session` | `hard_negative` |
| Erkin gapirish (uyg'otuvchi iborasiz) | `mobyww-record free` | `negative` |
| Sintetik (TTS) | W2, hali yo'q | `wake` / `negative` |
| Ochiq to'plamlar | faqat [licenses.md](licenses.md) ruxsat bergani | `negative`, `ambient`, `silence` |

Ro'yxat va sonlar `configs/wakeword.yaml` da. "mobil" va "mobile" o'zbek va rus tilida keng tarqalgan; ular hech qachon `wake` bo'lmaydi.

## Qadamlar

1. Ovoz egasi [maxfiylik hujjatini](privacy.md) o'qiydi. `mobyww-record consent` rozilik matnini ko'rsatadi; faqat `AGREE` yozilsa `data/consent/<taxallus>.json` yaratiladi. Taxallus tasodifiy (`spk-xxxxxx`), haqiqiy ism emas. Yozuv va rozilik git'da emas.
2. `mobyww-record plan` yozmasdan to'liq rejani chiqaradi (necha klip, qaysi masofa va ovoz).
3. `mobyww-record session --speaker spk-xxxxxx --language uz --room <xona> --device <mikrofon>`. Har taklif uchun Enter yozadi, `s` o'tkazadi, `q` to'xtatadi. Masofalar bo'yicha tartib: ovoz egasi kamroq joy almashtiradi. Mikrofon o'lik bo'lsa (jimlik) klip saqlanmaydi.
4. Boshqa xona yoki qurilma uchun sessiyani qayta ishga tushiring; xona va qurilma har klipning metadata'sida.
5. `mobyww-record free` erkin nutq bo'laklarini `negative` sifatida yozadi. Ovoz egasi uyg'otuvchi iborani aytmasligi kerak, aks holda klipni qo'lda olib tashlang.
6. Bo'lish: `mobyww.data.split.assign_splits` ovoz egasi bo'yicha deterministik bo'ladi (urug' `configs/wakeword.yaml`). Bitta odam ikki to'plamda bo'lmaydi. Test to'plamiga faqat yakuniy hisobotda tegiladi.
7. Tekshirish: `python -m mobyww.data.validate data/manifests/recorded.jsonl --root data --consent-dir data/consent`. Quyidagilarda yiqiladi: litsenziya yo'q, rozilik yo'q yoki qaytarib olingan, takroriy yozuv (kontent xeshi), ovoz egasining sizib o'tishi, noto'g'ri yo'l, xesh mos kelmasligi, taqsimlanmagan bo'lim.

## Manifest

Bir qator bir klip (JSONL). Sxema: [`data/manifest.schema.json`](../data/manifest.schema.json); majburiy maydonlar: `id`, `path`, `sha256`, `label`, `phrase`, `language`, `source`, `source_name`, `license`, `speaker`, `split`, `duration_s`, `sample_rate`. Yozilgan klipda qo'shimcha: `consent_id`, `distance_cm`, `volume`, `room`, `device`, `recorded_on`, `mode`. Yozish vositasi `split` ni `unassigned` qoldiradi, validator uni bo'lishdan keyingina qabul qiladi.

## Domen farqi (hujjatlashtirilgan xavf)

Boshida kompyuter yoki USB mikrofon ishlatiladi; robotda esa INMP441 (24 bit, 32 bitli slotda) bo'ladi. Ular balandlik, shovqin va chastota javobi bo'yicha farq qiladi. Ikki choralar: augmentatsiyada qurilma yo'lini taqlid qilish (W2, [augmentation.yaml](../configs/augmentation.yaml)) va keyinroq robotdan xom kadrlarni olish skripti bilan INMP441 yo'lidan real yozuvlar yig'ish. `device` maydoni har klipda bo'lgani uchun hisobotda qurilma bo'yicha ajratish mumkin. Bu yo'l hali yozilmagan.

## Cheklovlar va xavflar

- Hajm: rejadagi yig'ish bir necha ovoz egasi bilan kichik; aksent qamrovi (o'zbek, rus, ingliz) hozircha noma'lum. Natijalar test hajmi bilan birga beriladi.
- Yozish vositasi real mikrofon bilan hali sinalmagan (`sounddevice` yo'li testlarda yo'q).
