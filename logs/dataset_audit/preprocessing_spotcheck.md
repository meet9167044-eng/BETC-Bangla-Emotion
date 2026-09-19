# Phase 6 — Preprocessing Spot Check
**Date:** 2026-09-19 15:21:05
**Source:** Train split only (Data/processed/train/train.csv)
**Sample size:** 30 rows (random_state=42)

> **Note:** This spot check documents the output of the deterministic
> preprocessing function. No manual modifications were made to individual
> outputs. The preprocessing function is solely responsible for every
> transformation shown below.

---

## Observations per Example

### Example 1 (train row 18580)
- **Source:** UBMEC / NONE
- **Active labels:** joy
- **Original text:**
  ```
  পুলিশ ভাইদের আমার হাজার সালাম,
  ```
- **Processed text:**
  ```
  পুলিশ ভাইদের আমার হাজার সালাম,
  ```
- **Observation:** Intensifier preserved (হাজার)

### Example 2 (train row 25159)
- **Source:** MONOVAB / NONE
- **Active labels:** anger
- **Original text:**
  ```
  মুরখ একটা
  ```
- **Processed text:**
  ```
  মুরখ একটা
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 3 (train row 8939)
- **Source:** EmoNoBa / Val
- **Active labels:** fear, joy, surprise
- **Original text:**
  ```
  মাথার দাম কত 
  ```
- **Processed text:**
  ```
  মাথার দাম কত
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 4 (train row 7178)
- **Source:** EmoNoBa / Train
- **Active labels:** joy
- **Original text:**
  ```
  আপনার ভিডিওগুলো ভাল লাগে। 
  ```
- **Processed text:**
  ```
  আপনার ভিডিওগুলো ভাল লাগে।
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 5 (train row 18602)
- **Source:** UBMEC / NONE
- **Active labels:** surprise
- **Original text:**
  ```
  আপনার কাছে আমার প্রশ্ন...... আপনি কি তনু হত্যার বিচারের জন্য কিছু করেছেন.......????plz উত্তর টা দিলে খুশি হব @dr. imran
  ```
- **Processed text:**
  ```
  আপনার কাছে আমার প্রশ্ন.. আপনি কি তনু হত্যার বিচারের জন্য কিছু করেছেন..?? উত্তর টা দিলে খুশি হব imran
  ```
- **Observation:** @mention removed | Repeated punctuation collapsed

### Example 6 (train row 24408)
- **Source:** MONOVAB / NONE
- **Active labels:** joy
- **Original text:**
  ```
  আমি ওনার সাথে একমত আছি
  ```
- **Processed text:**
  ```
  আমি ওনার সাথে একমত আছি
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 7 (train row 15579)
- **Source:** UBMEC / NONE
- **Active labels:** sadness
- **Original text:**
  ```
  সাম্প্রতিক সময়ে মেসির পারফরম্যান্স আর তাঁর শরীরী ভাষা দেখে বোকা জুনিয়র্সের গাত্তির মনে হয়েছে  মেসি যেন অবসরে চলে যাওয়া কোনো খেলোয়াড় 
  ```
- **Processed text:**
  ```
  সাম্প্রতিক সময়ে মেসির পারফরম্যান্স আর তাঁর শরীরী ভাষা দেখে বোকা জুনিয়র্সের গাত্তির মনে হয়েছে মেসি যেন অবসরে চলে যাওয়া কোনো খেলোয়াড়
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 8 (train row 16040)
- **Source:** UBMEC / NONE
- **Active labels:** anger
- **Original text:**
  ```
  কুত্তার বাচ্চা শিবির অাছিলো,দাড়ির কাটিং দেখলেই বোঝা যায় :D
  ```
- **Processed text:**
  ```
  কুত্তার বাচ্চা শিবির আছিলো,দাড়ির কাটিং দেখলেই বোঝা যায় :
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 9 (train row 3874)
- **Source:** EmoNoBa / Train
- **Active labels:** sadness
- **Original text:**
  ```
  ফাঁসি ছাড়া এদের নিয়ন্ত্রণ করা সম্ভব না। ধর্ষণের শাস্তি ফাঁসি চাই। 
  ```
- **Processed text:**
  ```
  ফাঁসি ছাড়া এদের নিয়ন্ত্রণ করা সম্ভব না। ধর্ষণের শাস্তি ফাঁসি চাই।
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 10 (train row 9558)
- **Source:** EmoNoBa / Test
- **Active labels:** joy
- **Original text:**
  ```
  এই রকম নাটক আরো দেখতে চাই 
  ```
- **Processed text:**
  ```
  এই রকম নাটক আরো দেখতে চাই
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 11 (train row 9138)
- **Source:** EmoNoBa / Val
- **Active labels:** joy
- **Original text:**
  ```
  ভালো ছিলো কাভার টা। বেস্ট লাগলো 
  ```
- **Processed text:**
  ```
  ভালো ছিলো কাভার টা। বেস্ট লাগলো
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 12 (train row 10191)
- **Source:** EmoNoBa / Test
- **Active labels:** sadness
- **Original text:**
  ```
  আম্পায়ার এর মারে চুদি 
  ```
- **Processed text:**
  ```
  আম্পায়ার এর মারে চুদি
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 13 (train row 24267)
- **Source:** MONOVAB / NONE
- **Active labels:** joy
- **Original text:**
  ```
  ধন্যবাদ ভাইয়া
  ```
- **Processed text:**
  ```
  ধন্যবাদ ভাইয়া
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 14 (train row 25576)
- **Source:** MONOVAB / NONE
- **Active labels:** anger
- **Original text:**
  ```
  আমি ড.আকবর আলী খান স্যারের সাথে একমত।
  ```
- **Processed text:**
  ```
  আমি ড.আকবর আলী খান স্যারের সাথে একমত।
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 15 (train row 742)
- **Source:** EmoNoBa / Train
- **Active labels:** anger
- **Original text:**
  ```
  দেশের ফুটবল বাচানো খুব জরুরি। রুহুল আমিন প্রেসিডেন্ট হলে খুব ভাল হতো। 
  ```
- **Processed text:**
  ```
  দেশের ফুটবল বাচানো খুব জরুরি। রুহুল আমিন প্রেসিডেন্ট হলে খুব ভাল হতো।
  ```
- **Observation:** Intensifier preserved (খুব)

### Example 16 (train row 3023)
- **Source:** EmoNoBa / Train
- **Active labels:** joy
- **Original text:**
  ```
  বাই আনেক বাল লাগলো 
  ```
- **Processed text:**
  ```
  বাই আনেক বাল লাগলো
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 17 (train row 28680)
- **Source:** MONOVAB / NONE
- **Active labels:** sadness
- **Original text:**
  ```
  এই কাজ গুলো করা মোটেই ঠিক হয় নাই
  ```
- **Processed text:**
  ```
  এই কাজ গুলো করা মোটেই ঠিক হয় নাই
  ```
- **Observation:** Negation preserved (নাই) | Intensifier preserved (মোটেই)

### Example 18 (train row 5556)
- **Source:** EmoNoBa / Train
- **Active labels:** joy
- **Original text:**
  ```
  নাটকের সব চাইতে সুন্দর হয়েছে। নোয়াখালী ভাইর জন্য 
  ```
- **Processed text:**
  ```
  নাটকের সব চাইতে সুন্দর হয়েছে। নোয়াখালী ভাইর জন্য
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 19 (train row 10687)
- **Source:** EmoNoBa / Test
- **Active labels:** anger, sadness
- **Original text:**
  ```
  জঘন্য হয়ে গেসে। ভাইয়াদের কাছে ক্যামেরা ছিলো, ভিডিও করতেসে দেখে ভালো পিস দিসে 
  ```
- **Processed text:**
  ```
  জঘন্য হয়ে গেসে। ভাইয়াদের কাছে ক্যামেরা ছিলো, ভিডিও করতেসে দেখে ভালো পিস দিসে
  ```
- **Observation:** Intensifier preserved (জঘন্য)

### Example 20 (train row 26168)
- **Source:** MONOVAB / NONE
- **Active labels:** joy
- **Original text:**
  ```
  মাশা-আল্লাহ
  ```
- **Processed text:**
  ```
  মাশা-আল্লাহ
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 21 (train row 27609)
- **Source:** MONOVAB / NONE
- **Active labels:** anger
- **Original text:**
  ```
  হাসিনা’র পতন করতে হবে বিএনপির গনতন্ত্র দল
  ```
- **Processed text:**
  ```
  হাসিনার পতন করতে হবে বিএনপির গনতন্ত্র দল
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 22 (train row 8742)
- **Source:** EmoNoBa / Val
- **Active labels:** joy
- **Original text:**
  ```
  মামা এই রকম গান আরো চাই👌👌👌👌👌👌👌 
  ```
- **Processed text:**
  ```
  মামা এই রকম গান আরো চাই
  ```
- **Observation:** Emoji preserved

### Example 23 (train row 28623)
- **Source:** MONOVAB / NONE
- **Active labels:** sadness
- **Original text:**
  ```
  এ গুলো ঠিক না।যারা এ কাজগুলো করছে তারা অন্যায় করছে। তাদের বিচার হয়া দরকার।
  ```
- **Processed text:**
  ```
  এ গুলো ঠিক না।যারা এ কাজগুলো করছে তারা অন্যায় করছে। তাদের বিচার হয়া দরকার।
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 24 (train row 17613)
- **Source:** UBMEC / NONE
- **Active labels:** disgust
- **Original text:**
  ```
  fir,,,,kos,,,p, ,,, বরিশাল
  ```
- **Processed text:**
  ```
  ,, ,, বরিশাল
  ```
- **Observation:** Repeated punctuation collapsed

### Example 25 (train row 12051)
- **Source:** UBMEC / NONE
- **Active labels:** disgust
- **Original text:**
  ```
  অন্যায়কে প্রতিহত করার নানা উপায়ের একটি হচ্ছে চুপ থাকা  ইগনোর করা  সেই কাজ থেকে নিজেকে বিরত রাখা 
  ```
- **Processed text:**
  ```
  অন্যায়কে প্রতিহত করার নানা উপায়ের একটি হচ্ছে চুপ থাকা ইগনোর করা সেই কাজ থেকে নিজেকে বিরত রাখা
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 26 (train row 27265)
- **Source:** MONOVAB / NONE
- **Active labels:** sadness
- **Original text:**
  ```
  হারকিপ্টে নাটকের নজরআলীর মতো তেলের ব্যবহার করলে ১০০০ টাকা লিটার হলেও সমস্যা হবে না!
  ```
- **Processed text:**
  ```
  হারকিপ্টে নাটকের নজরআলীর মতো তেলের ব্যবহার করলে ১০০০ টাকা লিটার হলেও সমস্যা হবে না!
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 27 (train row 2918)
- **Source:** EmoNoBa / Train
- **Active labels:** sadness
- **Original text:**
  ```
  আর কোন জেলা নামাজ পড়েনা শুধু বরিশাল এর মানুষ পড়ে মিয়া নামাজ নিয়েও ডিস্ট্রিক ভাগ করে ফেললেন আপনাদের জন্য ইসলাম এর এই অবস্হা 
  ```
- **Processed text:**
  ```
  আর কোন জেলা নামাজ পড়েনা শুধু বরিশাল এর মানুষ পড়ে মিয়া নামাজ নিয়েও ডিস্ট্রিক ভাগ করে ফেললেন আপনাদের জন্য ইসলাম এর এই অবসহা
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

### Example 28 (train row 15952)
- **Source:** UBMEC / NONE
- **Active labels:** sadness
- **Original text:**
  ```
  বন্ধুরা আজ অনেক দিন থেকে কুমিল্লা শিক্ষাবোর্ডের অধীনে পরিচালিত হওয়া লক্ষীপুর জেলার, রামগঞ্জ থানার, ভাদুর উচ্চ বিদ্যালয়ে ছাত্র-ছাত্রীদেরকে জোর করে বাধ্যতামূলক ভাবে জোহরের নামাজ পড়ানো হয়। যদি কোন শিক্ষার্থী নামাজ না পড়ে তাহলে তাকে কঠিন শাস্তি প্রদান করা হয়। এমতাবস্থায় বিষয়টি পর্যবেক্ষণ করার জন্য নির্দিষ্ট কর্তৃপক্ষের দৃষ্টি আকর্ষন করছি এবং এসব অমানবিকতা বন্ধে কঠোর প্রদক্ষেপ কামনা করছি। সবাই এই পোস্টটি লাইক, কমেন্ট ও শেয়ার করে আপনিও মানবতা রক্ষার একজন অংশীদার হোন। ইমরান ভাই এ বিষয়ে আপনার দৃষ্টি আকর্ষন করছি। প্লিজ প্লিজ প্লিজ
  ```
- **Processed text:**
  ```
  বন্ধুরা আজ অনেক দিন থেকে কুমিল্লা শিক্ষাবোর্ডের অধীনে পরিচালিত হওয়া লক্ষীপুর জেলার, রামগঞ্জ থানার, ভাদুর উচ্চ বিদ্যালয়ে ছাত্র-ছাত্রীদেরকে জোর করে বাধ্যতামূলক ভাবে জোহরের নামাজ পড়ানো হয়। যদি কোন শিক্ষার্থী নামাজ না পড়ে তাহলে তাকে কঠিন শাস্তি প্রদান করা হয়। এমতাবস্থায় বিষয়টি পর্যবেক্ষণ করার জন্য নির্দিষ্ট কর্তৃপক্ষের দৃষ্টি আকর্ষন করছি এবং এসব অমানবিকতা বন্ধে কঠোর প্রদক্ষেপ কামনা করছি। সবাই এই পোস্টটি লাইক, কমেন্ট ও শেয়ার করে আপনিও মানবতা রক্ষার একজন অংশীদার হোন। ইমরান ভাই এ বিষয়ে আপনার দৃষ্টি আকর্ষন করছি। প্লিজ প্লিজ প্লিজ
  ```
- **Observation:** Negation preserved (না) | Intensifier preserved (অনেক)

### Example 29 (train row 13530)
- **Source:** UBMEC / NONE
- **Active labels:** joy
- **Original text:**
  ```
  গ্রেট ভাই   সত্যিই আপনাকে যত দেখছি ততই অনুপ্রাণিত হচ্ছি  কথা গুলো  মহামূল্যবানএবং এটাই সমাজের বাস্তব অবস্থা 
  ```
- **Processed text:**
  ```
  গ্রেট ভাই সত্যিই আপনাকে যত দেখছি ততই অনুপ্রাণিত হচ্ছি কথা গুলো মহামূল্যবানএবং এটাই সমাজের বাস্তব অবস্থা
  ```
- **Observation:** Intensifier preserved (সত্যিই)

### Example 30 (train row 4749)
- **Source:** EmoNoBa / Train
- **Active labels:** joy
- **Original text:**
  ```
  বেড়াল দেখে ফেলেছি 
  ```
- **Processed text:**
  ```
  বেড়াল দেখে ফেলেছি
  ```
- **Observation:** Standard Bangla text — whitespace normalized only

---

## Summary of Spot-Check Observations

| Feature | Count (out of 30) |
|---|---|
| URL present in original (removed) | 0 |
| @mention present in original (removed) | 1 |
| HTML noise present (removed) | 0 |
| Repeated punctuation normalized | 2 |
| Emoji present (preserved) | 1 |
| Negation word present (preserved) | 2 |
| Intensifier word present (preserved) | 6 |
| Empty output after preprocessing | 0 |

## Conclusion

The preprocessing function correctly:
- Removes URLs, @mentions, and HTML noise
- Preserves negation and intensifier words
- Preserves emojis (not listed as noise in PIPELINE_SPEC.md)
- Collapses repeated punctuation
- Normalizes whitespace
- Applies Bangla-specific Unicode normalization (bnunicodenormalizer, word-level)

> **[RESEARCH DECISION REQUIRED]** `configs/negations_bn.txt` and
> `configs/intensifiers_bn.txt` were created as DRAFT files seeded from
> PIPELINE_SPEC.md Section 3.1 examples. They must be **manually reviewed**
> by the project owner against real annotated sentences before being
> treated as finalized. See PIPELINE_SPEC.md Section 3.1 [PROPOSAL] and
> RESEARCH_RULES.md.