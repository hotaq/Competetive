# Kaggriculture Discussions Research — 22 สิงหาคม 2026

สถานะ: อ่านจาก Kaggle Discussions API ทางการครบ 6 หน้า รวม 103 หัวข้อ ณ วันที่ 22 สิงหาคม 2026 และคัดอ่านเฉพาะหัวข้อที่เกี่ยวกับคะแนน, market, validation, replay, RL และกลยุทธ์คะแนนสูง

ขอบเขตหลักฐาน: ใช้เฉพาะหน้า Discussion และ API ของ Kaggle เป็นแหล่งต้นทางในรายงานนี้ ข้อความจากผู้จัดถือเป็นข้อมูลทางการ ส่วนผลทดลองหรือคำอธิบายของผู้แข่งขันเป็นหลักฐานเชิงประจักษ์ที่ยังไม่ได้รับรองโดย Kaggle และข้อเสนอสำหรับ V16/RL ถูกระบุเป็น “ข้ออนุมานของเรา” แยกต่างหาก

## ข้อสรุปสำหรับทิศทางของเรา

แนวทางที่มีโอกาสดีที่สุดคือเก็บ V16 เป็น deterministic executor ที่ผ่านการทดสอบ แล้วเพิ่มสองชั้นแบบปิดวงจร:

1. adaptive market policy ที่อ่านราคา, inventory, shop mix, day และพฤติกรรมขายของคู่แข่งจริง แทนการผูกการขายกับ turn คงที่
2. RL meta-controller ที่เลือก macro/route หรือสั่ง deviation จำนวนเล็กน้อย โดย deterministic executor ยังดูแล legality, movement และงานฟาร์มระดับต่ำ

ไม่ควรเริ่มจาก end-to-end PPO ควบคุมทุก action ทุก turn: รายงานล่าสุดหลายชุดพบ cold start, action-space และ long-horizon credit assignment เป็นคอขวด ขณะที่การทดลอง meta-agent พบ payoff cycle ชัดเจน จึงควรวัดด้วย cross-play matrix และ worst-opponent win rate ไม่ใช่ค่าเฉลี่ยเหรียญหรือคู่แข่งตัวเดียว

## 1. Engine และ balance patch ที่ต้องยึดเป็นฐาน

### ข้อมูลทางการจากผู้จัด

- วันที่ 6 สิงหาคม ผู้จัดประกาศในหัวข้อ `733431` ว่า Town Center ลดการซื้อเหลือสินค้า 1 ชิ้นต่อวันและไม่เพิ่ม multiplier ช่วงท้ายเกม อีกทั้ง shop ถูกสุ่มแบบ **with replacement** ทำให้หนึ่งเกมอาจมี shop ชนิดเดียวซ้ำหลายแห่งหรือไม่มีบางชนิดเลย พร้อมกำหนดให้ใช้ `kaggle-environments >= 1.32.6` ([Balance Changes](https://www.kaggle.com/competitions/kaggriculture/discussion/733431), 2026-08-06)
- วันที่ 15 สิงหาคม ผู้จัดประกาศในหัวข้อ `735311` ว่า scarcity-price ของ tomato, carrot และ egg จะเพิ่มแรงขึ้นเมื่อมี shop demand มากแต่ไม่มี production โดยกรณี no-production คาดว่าจะเกิดประมาณ 50%, 26% และ 22% ตามลำดับ และกำหนดให้ใช้ `>= 1.32.7` ผู้จัดระบุด้วยว่านี่ควรเป็น balance change สุดท้ายเว้นแต่พบ game-breaking bug ([Small balance change](https://www.kaggle.com/competitions/kaggriculture/discussion/735311), 2026-08-15)
- ในหัวข้อ `732450` ผู้จัดตอบยืนยันว่า engine เป็น source of truth และยืนยันรายละเอียดสำคัญ ได้แก่ care bonus จริงเป็น `+1`, fertilizer ขายได้, `DIG` ใช้กับโครงสร้างว่างเท่านั้น, พืชที่เพิ่งปลูกต้องรดน้ำในวันเดียวกัน, strawberry/tomato ให้ผลหลายครั้งแต่ไม่อนันต์ และ shed access อยู่ที่สี่พิกัดกลาง นอกจากนี้ปัญหา hand เกิดบน locked tile ถูกระบุว่าเป็นบั๊กและแก้แล้ว ([Documentation vs. Engine Discrepancies](https://www.kaggle.com/competitions/kaggriculture/discussion/732450), โพสต์ 2026-08-03; คำตอบผู้จัด 2026-08-04)

### ข้ออนุมานของเราสำหรับ V16

- pin simulator และ regression suite ที่ `1.32.7` และแยก replay ตาม `module_version`; ห้ามรวมผลก่อน/หลัง patch เป็น distribution เดียว
- gate ที่ผูกกับ step 408/456 ของ RC6 ควรเป็น safety fallback ไม่ใช่ market policy หลัก เพราะ shop mix และ scarcity regime เปลี่ยนตาม seed
- เพิ่ม engine-parity tests ครอบคลุม care, same-day watering, finite ongoing crops, fertilizer sale, structure digging, shed coordinates และ hand spawning

## 2. Market: “ของราคาแพง” ไม่เท่ากับ “ตลาดใหญ่”

### สิ่งที่ผู้แข่งขันวัดได้

- หัวข้อ `734412` วิเคราะห์จาก engine ว่าเมืองค่อย ๆ ขุด demand hole ตลอด season และการขายลงใน hole ตอนปลายเกมให้ผลต่างจากการขายเร็วมาก ผู้เขียนคำนวณ expected town/shop drain บน 1.32.6 เช่น wheat 525, strawberry 426, milk/carrot 327, wool/tomato/egg 228 และ melon 30; melon ไม่มีอยู่ในเมนู shop ทั้งแปดชนิดเลย ผู้เขียนรายงานด้วยว่าการลด movement thrashing จาก 83% เหลือ 55% ของ unit-turns ทำให้ bank ปลายเกมเพิ่มราวสามเท่า ([The price table ranks crops backwards](https://www.kaggle.com/competitions/kaggriculture/discussion/734412), 2026-08-11)
- ความคิดเห็นในหัวข้อเดียวกันให้การตรวจอิสระต่อ drain counts แต่ย้ำว่า shop draw แบบ replacement ทำให้ค่าเฉลี่ยหลอกได้: ตัวอย่างเช่น wool มีโอกาสไม่มี buyer และ carrot มี distribution กว้าง ผู้แสดงความคิดเห็นยังรายงานว่าหลัง 1.32.7 carrot ได้ upside สูงกว่า tomato/egg ใน tail ที่วัด แต่ egg แทบไม่ถึง scarcity ระดับทำเงินใน percentile ส่วนใหญ่ นี่เป็นผลทดลองของผู้แข่งขัน ไม่ใช่การรับรองจากผู้จัด ([ความคิดเห็นในหัวข้อ `734412`](https://www.kaggle.com/competitions/kaggriculture/discussion/734412), 2026-08-17)
- ความคิดเห็นในประกาศ 1.32.7 รายงานจาก 120 replay ว่า patch ไม่เปลี่ยนผู้ชนะเลยใน 224 paired comparisons แต่เพิ่ม tail dispersion บาง seed ผู้เขียนจึงเสนอให้มอง `Pr(win)` มากกว่า expected money ผลนี้เป็นการวัดของผู้แข่งขันและ sample-specific ([Small balance change](https://www.kaggle.com/competitions/kaggriculture/discussion/735311), ความคิดเห็น 2026-08-17)

### ข้ออนุมานของเราสำหรับ V16

ให้ market layer ตัดสินใจจาก live features ต่อไปนี้:

- `market.inventory`, current marginal sell price และ recent inventory delta ต่อสินค้า
- จำนวนและชนิด shop ที่เปิดจริง รวมถึง demand ที่เหลือ
- inventory ของเรา, production pipeline, day และเวลาที่เหลือก่อน terminal
- สัญญาณการขาย/production ของคู่แข่ง เพราะ market hole ใช้ร่วมกัน

การทดลองต้องเปรียบเทียบ “ขายตอนนี้ / รอ / เปลี่ยนสินค้า” ด้วย paired seed และทั้งสอง seat โดย optimize win probability และ worst-case ไม่ใช่ revenue เฉลี่ยอย่างเดียว

## 3. คะแนนสด, matchmaking และ final evaluation

### ข้อมูลทางการ

- ผู้จัดระบุในหัวข้อ `731587` ว่าหลัง submission deadline ระบบจะปล่อยให้เกิด episodes ต่ออีกประมาณสองสัปดาห์ แล้วรัน **Bradley–Terry Tournament ครั้งเดียว** เพื่อกำหนด final leaderboard จุดประสงค์คือให้ hot streak มีผลน้อยลง ([Comment on final evaluation](https://www.kaggle.com/competitions/kaggriculture/discussion/731587), 2026-07-31)

### รายงานของผู้แข่งขัน — ไม่ใช่สเปกทางการ

- ผู้แข่งขันรายงานในหัวข้อ `734000` ว่า agent สองไฟล์ที่ byte-identical และส่งห่างกันราวสองชั่วโมงไปอยู่ประมาณ 1700 กับมากกว่า 3000 ต่างกันราว 1400 คะแนน จึงตั้งข้อสังเกตว่า live rating มี path dependence สูง ความคิดเห็นอื่นรายงาน divergence ของ duplicate submission เช่นกัน ([How path dependent is current leaderboard rating?](https://www.kaggle.com/competitions/kaggriculture/discussion/734000), 2026-08-09)
- Ryo Hasegawa ระบุชัดว่าโพสต์ของตนเป็น fit และ assumption จาก submission ของตน ไม่ใช่ข้อมูลภายใน Kaggle เขาประมาณว่า submission ได้ระดับราว 90% หลังประมาณ 60 เกม, residual noise ราว ±25–50, ควรเทียบ agent ที่ game count ใกล้กัน และควร optimize win rate ต่อ strong opponents ด้วย seed เดียวกัน แทน coin margin เฉลี่ย ([1st Place Currently — Submission Strategy](https://www.kaggle.com/competitions/kaggriculture/discussion/736219), 2026-08-19)

### ข้ออนุมานของเราสำหรับ validation

- live public score ใช้เป็น smoke signal ไม่ใช่ objective หลัก
- ทุก candidate ต้องผ่าน paired seeds, both seats และ cross-play matrix แยกตาม opponent
- metric หลักควรมี per-opponent win rate, aggregate win rate, worst-opponent win rate, worst-case/low-quantile margin และ regression count
- submission ใหม่ต้องเป็นการเปลี่ยนที่ผ่าน offline evidence และ entrypoint/runtime smoke test แล้ว; หลีกเลี่ยงการ re-roll ไฟล์เดิมเพื่อลุ้น rating สด

## 4. หลักฐานเรื่อง RL และ payoff ที่ไม่เป็นลำดับเส้นตรง

### End-to-end RL

- ผู้แข่งขันที่ลองให้โมเดลควบคุมทุก action ทุก turn รายงานว่า BC เรียนได้เพียงรูปแบบซ้ำ ๆ จาก replay, exploration ยากเพราะ action space ใหญ่และ crop cycles ยาว และ harvest success บน plot ที่สอง/สามอยู่เพียงประมาณ 20–40% จน policy เรียนว่าไม่ควรซื้อที่ดิน นี่เป็นรายงานหนึ่งทีม ไม่ใช่ข้อพิสูจน์ทั่วไป ([End-to-End RL Is Harder Than It Looks](https://www.kaggle.com/competitions/kaggriculture/discussion/736567), 2026-08-21)
- ในหัวข้อ self-play ผู้ใช้รายหนึ่งรายงาน pure self-play จบที่ 20k–22k gold ขณะที่อีกรายรายงาน hybrid rule-based + PPO ถึง 100k แต่ไม่มี artifact หรือ protocol ที่ยืนยันตัวเลขใน thread จึงใช้ได้เพียงเป็น anecdote ว่า hybrid น่าทดลอง ไม่ใช่ benchmark ([Is Pure Self-Play PPO viable?](https://www.kaggle.com/competitions/kaggriculture/discussion/734952), 2026-08-13)
- หัวข้อถามผู้แข่งขันอันดับสูงมีคำตอบหลายคนว่า submission ปัจจุบันยังเป็น heuristic และมีผู้ใช้รายหนึ่งระบุว่า heuristic ไปถึง 2400+ ได้ แต่ thread ไม่ได้พิสูจน์ว่าไม่มีทีมอื่นใช้ RL ([Any High-Ranking Entries Using RL?](https://www.kaggle.com/competitions/kaggriculture/discussion/736369), 2026-08-20 ถึง 2026-08-22)

### Meta-agent และ non-transitive matchups

- การทดลอง meta-agent ที่ให้ PPO เลือกระหว่าง high-scoring public trajectories ใช้ 960 เกม, 6 strategies, 16 seeds และทั้งสอง seats พบ cycle ชัดเจน: Public B85 ชนะ Andrews 2883 `30–2`, Andrews ชนะ Kaito v35 `21–11`, และ Kaito ชนะ Public B85 `24–8` แม้ Adaptive Farming จะดีที่สุดโดยรวมที่ `121/160` แต่ยังแพ้ Kaito v35 `15–17` ([RL Meta Agent: 960 Matchups PPO Plateau](https://www.kaggle.com/competitions/kaggriculture/discussion/736439), 2026-08-20)
- ใน diagnostic เดียวกัน `opponent + opening branch` ทำนาย outcome ได้ 67.5% เทียบกับ opponent-only และ opening-only อย่างละ 56.5% ผู้เขียนตีความว่า PPO gradients อาจขัดกันเมื่อ route ที่ดีต่อคู่แข่งหนึ่งแพ้อีกตัว และ policy ต้องเลือก opening ก่อนเห็นข้อมูลพอ นี่เป็น hypothesis ที่สอดคล้องกับ payoff matrix แต่ยังไม่พิสูจน์ equilibrium ([หัวข้อ `736439`](https://www.kaggle.com/competitions/kaggriculture/discussion/736439))

### ข้ออนุมานของเราสำหรับ RL design

- action space แรกควรเป็น macro/route/deviation ที่มี mask ไม่ใช่ low-level 720-turn control
- freeze V16 executor ระหว่างฝึก meta-policy เพื่อให้ reward attribution ไม่เจอกับ moving target
- train กับ opponent league และ historical variants; ห้าม train เฉพาะ current top-1
- ใช้ payoff matrix และ robust mixture objective เพราะ average win rate อาจซ่อน hard counter
- หากต้องเลือก opening ก่อนรู้คู่แข่ง ให้ใช้ mixture ที่คุม worst case แล้วค่อยอนุญาต route switch เมื่อ observation มี evidence เพียงพอ

## 5. Replay corpus และความเสี่ยงจากข้อมูลลำเอียง

- ผู้จัดทำ Daily Top Episodes โดยเรียง episodes ตาม average rating ของ agent ณ เวลานั้นและเผยแพร่สูงสุดประมาณ 20 GB ต่อวัน เพื่อสนับสนุน IL/BC/RL ([Daily Top Episodes Dataset](https://www.kaggle.com/competitions/kaggriculture/discussion/731215), 2026-07-31)
- ผู้แข่งขันอีกทีม intersect Daily Top dumps 22 ชุด (16,031 unique episodes) กับ API-crawled corpus 47,351 episodes และรายงานว่า Aug 9–20 จำนวน 8,300 dump episodes มี overlap เป็นศูนย์กับ corpus ของตน เขาย้ำว่าทั้งสองแหล่งยังเป็น biased slice ของ ladder และ replay ต้องแบ่งตาม `module_version` ผลนี้ตรวจเฉพาะ corpus ของผู้เขียน ไม่รับประกันกับ crawler อื่น ([Daily Top Episodes dumps stopped overlapping](https://www.kaggle.com/competitions/kaggriculture/discussion/736625), 2026-08-21)

### ข้ออนุมานของเรา

- รวม official daily dumps กับ API corpus แต่ deduplicate ด้วย episode ID
- stratify ด้วย engine version, opponent family, seat, shop mix และ date/meta generation
- split train/validation ด้วย episode ID และ opponent family ไม่ใช่สุ่ม transition เพื่อหลีกเลี่ยง leakage
- อย่า BC จาก top trajectory family เดียว เพราะจะเลียนแบบ opening ซ้ำและแพ้ hard counter

## 6. Shared high-score meta และ loop พัฒนาที่นำมาใช้ได้

- วันที่ 9 สิงหาคม ผู้แข่งขันวิเคราะห์ว่า 3 ใน top-5 ใช้ opening ตรงกันกับ public reference agent และอยู่ในช่วง rating 3,117–3,131 ขณะที่สองอันดับบนใช้ opening ที่ไม่ตรงกับ public notebook ข้อมูลนี้เป็น snapshot เก่าและไม่ควรตีความว่าเป็น leaderboard ปัจจุบัน ([Analyzed top-5's openings](https://www.kaggle.com/competitions/kaggriculture/discussion/733924), 2026-08-09)
- Kaito Fukami อธิบาย loop ว่า “real losses → isolate one failure → build challengers → test multiple teams and both seats → reject most → freeze winner → test later episodes” และเก็บ representative strategies จาก meta รุ่นก่อน เพราะการ optimize เฉพาะ Top-30 ล่าสุดอาจแพ้ agent เก่าที่ยังอยู่ เขาย้ำด้วยว่าการ copy public meta ช่วยไต่อันดับได้เร็ว แต่ edge จะถูก meta ดูดซึม ([How I Keep Iterating Kaggriculture Agents](https://www.kaggle.com/competitions/kaggriculture/discussion/734212), 2026-08-10)
- โพสต์ “1st Place Currently” ที่คะแนนโหวตสูงสุดในชุดล่าสุดเปิดเผย mechanics/assumption เรื่อง rating แต่ไม่ได้เปิดเผย policy หรือ source ของ agent อันดับหนึ่ง จึงยังไม่มีหลักฐานจาก Discussions ที่ยืนยัน “สูตร top-1 ปัจจุบัน” ให้คัดลอก ([หัวข้อ `736219`](https://www.kaggle.com/competitions/kaggriculture/discussion/736219), 2026-08-19)

## 7. ลำดับทดลองที่แนะนำ

1. **P0 — Kaggle parity:** pin 1.32.7, เพิ่ม callable/entrypoint smoke test และ engine-mechanics regression tests
2. **P1 — Adaptive market V16:** แทน fixed sale timing ด้วย state-based gate แล้ววัด paired cross-play ต่อ opponent หลาย family
3. **P1 — Robust evaluator:** ทำ payoff matrix, both seats, held-out seeds, worst-opponent win rate และ no-regression gate ก่อน submit
4. **P2 — RL meta-controller:** เริ่มจากเลือก macro/deviation เหนือ frozen V16 executor พร้อม opponent league และ action masks
5. **P2 — Replay pipeline:** รวม/deduplicate official dumps กับ API corpus, stratify ตาม version/meta และกัน family leakage
6. **P3 — Closed-loop expansion:** ให้ meta-controller switch route เมื่อเห็น opponent/shop evidence; เทียบกับ robust static mixture ก่อนเพิ่ม model complexity

## ข้อจำกัด

- Discussion bodies บางโพสต์เป็นประสบการณ์หรือ fit ส่วนบุคคล และไม่มี code/data artifact อยู่ในข้อความ จึงไม่ใช้เป็นข้อเท็จจริงทางการ
- ตัวเลข leaderboard และ “top” ทุกตัวเป็น snapshot ตามวันที่โพสต์; meta อาจเปลี่ยนหลังจากนั้น
- รายงานนี้ไม่ได้ตรวจ Kaggle Code notebooks ภายนอก Discussions ตามข้อจำกัดของงาน และไม่พบ discussion ที่เปิดเผย source ของ top-1 ปัจจุบันอย่างตรวจสอบได้
