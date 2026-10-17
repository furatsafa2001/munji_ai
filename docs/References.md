# References — المراجع المُعتمَدة

قائمة المراجع التي بُنيت عليها قرارات نموذجَي `shockable_net` و `rhythm_net`.
كل رابط تُحقِّق منه.

---

## SHOCKABLE — نموذج السكتة القلبية

### [1] معيار أداء خوارزميات نصح الصدمة — المصدر الأصلي

Kerber RE, Becker LB, Bourland JD, Cummins RO, Hallstrom AP, Michos MB,
Nichol G, Ornato JP, Thies WH, White RD, Zuckerman BD.
**Automatic External Defibrillators for Public Access Defibrillation:
Recommendations for Specifying and Reporting Arrhythmia Analysis Algorithm
Performance, Incorporating New Waveforms, and Enhancing Safety.**
A statement for health professionals from the American Heart Association Task
Force on Automatic External Defibrillation, Subcommittee on AED Safety and
Efficacy.
*Circulation.* 1997;95(6):1677–1682.
DOI: 10.1161/01.CIR.95.6.1677

https://www.ahajournals.org/doi/full/10.1161/01.CIR.95.6.1677
PubMed: https://pubmed.ncbi.nlm.nih.gov/9118556/

**استُخدم في:** الإطار المرجعي لتقييم أداء الخوارزمية.

---

### [2] عتبات AHA الرقمية

Figuera C, Irusta U, Morgado E, Aramendi E, Ayala U, Wik L, Kramer-Johansen J,
Eftestøl T, Alonso-Atienza F.
**Machine Learning Techniques for the Detection of Shockable Rhythms in
Automated External Defibrillators.**
*PLOS One.* 2016.

https://pmc.ncbi.nlm.nih.gov/articles/PMC4956226/

**ما ورد فيه:** توصية AHA بحساسية أعلى من ٩٠٪ للإيقاعات القابلة للصدمة،
ونوعية أعلى من ٩٥٪ لغير القابلة للصدمة، وأعلى من ٩٩٪ للإيقاع الجيبي الطبيعي.

**استُخدم في:**
```python
SENSITIVITY_FLOOR = 0.90   # munji_ai/models/shockable_net.py
```

---

### [3] عتبة ١٥٠ نبضة/دقيقة للتسرّع البطيني السريع

Krasteva V, Jekova I.
**Shock advisory tool: Detection of life-threatening cardiac arrhythmias and
shock success prediction by means of a common parameter set.**
*Biomedical Signal Processing and Control.* 2007;2(1):25–33.

https://www.sciencedirect.com/science/article/abs/pii/S1746809407000110

**ما ورد فيه:** أن أغلب المصنّعين يحددون التسرّع البطيني السريع بتردد
أعلى من ١٥٠ نبضة في الدقيقة. وأن توصيات AHA تُلزم بكشف الرجفان البطيني
الخشن والتسرّع البطيني السريع، وتمنع منعًا باتًّا تصنيف الإيقاع الجيبي
الطبيعي وباقي اضطرابات النظم المصحوبة بنبض محسوس على أنها قابلة للصدمة.

**استُخدم في:**
```python
VT_FAST_BPM = 150.0
SHOCKABLE_RHYTHMS = ("VF", "VT_FAST")
```

---

### [4] تحذير تفاؤل البيانات العامة

Picon A, Irusta U, Álvarez-Gila A, Aramendi E, Alonso-Atienza F, Figuera C,
Ayala U, Garrote E, Wik L, Kramer-Johansen J, Eftestøl T.
**Mixed convolutional and long short-term memory network for the detection of
lethal ventricular arrhythmia.**
*PLOS One.* 2019;14(5):e0216756.

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0216756
PMC: https://ncbi.nlm.nih.gov/pmc/articles/PMC6527215

**ما ورد فيه:** أن كشف الإيقاعات القابلة للصدمة أسهل على بيانات المستودعات
العامة منه على بيانات السكتة القلبية خارج المستشفى، وأن هذه النتيجة سبق
إثباتها بخوارزميات التعلم الآلي الكلاسيكية. والسبب أن بيانات السكتة الحقيقية
تُسجَّل في بيئة ما قبل المستشفى، وبحلول وصول خدمات الطوارئ يكون العضل القلبي
قد تدهور بدرجة كبيرة.

**كما ورد فيه:** أن التقسيم أُجري على مستوى المريض — ٨٠٪ للتدريب و٢٠٪ للتقييم،
وعلى كل قاعدة على حدة قبل الدمج.

**استُخدم في:** التحذير المكتوب في رأس `munji_ai/train/train_shockable.py`،
ومنهج التقسيم على مستوى المريض.

---

### [5] مفتاح توسيم قاعدة VFDB — أهم مرجع عملي

PhysioNet / PhysioBank.
**Annotations in the MIT-BIH Malignant Ventricular Arrhythmia Database.**

https://archive.physionet.org/physiobank/database/vfdb/annotations.shtml

**ما ورد فيه:** أن جميع التوسيمات في ملفات `.atr` هي توسيمات تغيّر إيقاع،
وجدول كامل لتفسير سلاسل `aux` — من بينها `(VF` أو `(VFIB` للرجفان البطيني،
و`(VFL` للرفرفة البطينية، و`(VT` للتسرّع البطيني، و`(NOISE` للضوضاء،
و`(ASYS` لتوقف القلب.

**والقاعدة الحاسمة:** أن توسيمات تغيّر الإيقاع تُوضع في **بداية** النوبة،
وأن **الإيقاع السابق يستمر** خلال النوبات المعلَّمة بالضوضاء.

**استُخدم في:** كل منطق قراءة التوسيم في `munji_ai/data/loader.py` —
وتحديدًا التعامل مع `(NOISE` كفئة غامضة لا كتغيّر إيقاع.

---

### [6] معيار WFDB و ANSI/AAMI EC38

PhysioNet / PhysioBank.
**MIT-BIH Arrhythmia Database CD-ROM (Third Edition) — Readme.**

https://archive.physionet.org/physiobank/database/readme.htm

**ما ورد فيه:** أن مجلدات `mitdb` و `cudb` و `nstdb` تحوي إشارات رقمية
وتوسيمات مرجعية يجب استخدامها لاختبار محلّلات ECG وفق المعيار الأمريكي
الوطني الحالي لأجهزة تخطيط القلب المتنقّلة (ANSI/AAMI EC38).

**استُخدم في:** تفسير رموز الأحداث `[` و `]` في CUDB، ومرجعية معيار EC38
لخطة الاختبار.

---

### [7] ANSI/AAMI DF80

Association for the Advancement of Medical Instrumentation.
**DF80 — Medical electrical equipment. Part 2-4: Particular requirements for
the basic safety and essential performance of cardiac defibrillators.**

https://www.aami.org

⚠️ **معيار مدفوع — لا رابط مفتوح.** العتبات الرقمية منه متاحة عبر المرجع [2].

---

## RHYTHM — نموذج الإيقاع

### [8] وصف قاعدة NSRDB

Mousavi S, Afghah F, Razi A, Acharya UR.
**ECGNET: Learning where to attend for detection of atrial fibrillation with
deep visual attention.**
*arXiv:1812.07422.*

https://arxiv.org/pdf/1812.07422

**ما ورد فيه:** أن قاعدة الإيقاع الجيبي الطبيعي تتكوّن من **١٨ تسجيلًا
طويل المدى لأشخاص لا يعانون اضطرابات نظم ذات دلالة، رجالًا ونساءً
بأعمار ٢٠ إلى ٥٠ سنة**. وأن قاعدة AF فيها ٢٥ تسجيلًا وقاعدة AF طويلة المدى
فيها ٨٤ تسجيلًا.

**كما ورد فيه:** أن الباحثين استخرجوا مقاطع **٣٠ ثانية** من هذه القواعد،
واعتمدوا توسيمات `(AFIB` لاختيار مقاطع الرجفان و`(N` لاختيار غير الرجفان.
وأنهم استخدموا عتبة نقاء **٥٠٪** لتسمية المقاطع.

**استُخدم في:**
```python
baseline_rhythm = "(N"   # munji_ai/data/registry.py — لـ nsrdb فقط
```

**وسند إضافي لـ:** طول نافذة ٣٠ ثانية، واستخدام NSRDB كمصدر للفئة السلبية.

---

## مراجع لم يتيسّر توثيق روابطها

⚠️ **دراسة SVM/GA لطول النافذة** — وردت في بحث أولي وأفادت أن تجربة أطوال
نوافذ من ١ إلى ١٠ ثوانٍ أعطت أفضل نتيجة عند ٥ ثوانٍ. **لم أتمكّن من استرجاع
الاستشهاد الكامل ولا الرابط.**

**التوصية:** إما البحث عنها وتوثيقها، أو استبدالها بالمرجع [3] الذي يغطي
نافذة التحليل ضمن سياق AED، أو حذف الادعاء المرتبط بها من التوثيق.
**رقم `WINDOW_SEC["shockable"] = 5.0` كان في `config.py` قبل هذا العمل**،
فالمرجع سند لاحق لا مصدر.

---

## المصدر غير المنشور

### [9] قالب المعمارية الداخلي

`munji_ai/models/quality_gate.py` — داخل المستودع.

منه: بنية `ConvBlock`، استخدام التوسعة (dilation)، التجميع العام بدل
التسطيح، وأسلوب سكربت التدريب واختيار الحقبة.

⚠️ **هذا هو الأساس المعماري الفعلي لكلا النموذجين — لا ورقة منشورة.**

---

## ملاحظة على الصياغة

في أي تسليم، الصيغة الصحيحة:

✅ «العتبات مأخوذة من توصيات AHA/AAMI»
✅ «قواعد قراءة التوسيم مأخوذة من مفاتيح PhysioNet الرسمية»
✅ «البنية توافق تصميمًا منشورًا لشبكة محدودة الموارد»

❌ «المعمارية مبنية على ورقة [رقم]»
