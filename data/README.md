# data/

Bu papka git'da emas. Faqat shu fayl va `manifest.schema.json` kuzatiladi; qolgan hamma narsa `.gitignore` bilan yopilgan.

Bu yerda lokal ravishda turadi (hech qachon git'ga qo'shilmaydi):

- `recordings/<ovoz-egasi>/*.wav`: yo'naltiruvchi vosita bilan yozilgan klip'lar.
- `consent/<ovoz-egasi>.json`: rozilik yozuvlari va ovoz egasining shaxsi haqida lokal eslatma.
- `manifests/*.jsonl`: manifestlar (bir qator, bir klip).
- yuklab olingan ochiq to'plamlar (faqat `docs/licenses.md` ruxsat bergan to'plamlar).

Manifest sxemasi: [`manifest.schema.json`](manifest.schema.json). Tekshirish: `python -m mobyww.data.validate data/manifests/recorded.jsonl --root data --consent-dir data/consent --allow-unassigned`.

Maxfiylik qoidalari: [`docs/privacy.md`](../docs/privacy.md). Yig'ish tartibi: [`docs/data-collection.md`](../docs/data-collection.md).
