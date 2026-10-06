# แนวทาง RL ที่ยั่งยืนสำหรับ Kaggriculture

> สถานะ: research memo — 15 สิงหาคม 2026  
> ขอบเขตหลักฐาน: ใช้งานวิจัยต้นฉบับเท่านั้นสำหรับข้ออ้างภายนอก และแยกข้อเท็จจริงของโปรเจกต์ออกจากผลวิจัย

## ข้อสรุป

แนวทางที่มีเหตุผลที่สุดไม่ใช่การให้ RL ควบคุม primitive actions ทั้ง 720 turns และไม่ใช่การเพิ่ม PPO อีกชั้นบน fixed policy เดิม แต่คือ:

**state-aware macro options → masked conditional policy → deterministic executor**

จากนั้นฝึกตามลำดับ:

**macro BC → macro DAgger → recurrent masked PPO → opponent league**

จุดสำคัญคือโปรเจกต์มีองค์ประกอบหลายอย่างอยู่แล้ว ทั้ง 30-step macro environment, codec, masks, entity transformer, BC, DAgger และ PPO จึงไม่ควรสร้าง full-control stack ใหม่โดยไม่วิเคราะห์สาเหตุเดิม งานที่ควรทำก่อนคือเปลี่ยนความหมายของ macro action จาก “offset รอบกฎ v2.3” ให้เป็น “เป้าหมายเชิงเศรษฐกิจจาก state ปัจจุบัน”, เพิ่มข้อมูล state, จัด reward ให้ตรงเป้าการแข่งขัน และขยาย opponent pool

## สิ่งที่มีอยู่และช่องว่างจริง

ข้อมูลจากการทดลองในโปรเจกต์ปัจจุบัน:

> หมายเหตุด้านหลักฐาน: ตอนทำ memo นี้ checkout ปัจจุบันไม่มี source implementation ชุด `rl/` ให้เปิดตรวจโดยตรง ข้อเท็จจริงเฉพาะโปรเจกต์ด้านล่างจึงมาจากดัชนี `.tokensave` และบันทึกผลทดลองเดิม ต้องตรวจซ้ำกับ source จริงก่อนเริ่มแก้โค้ด

| สิ่งที่มีอยู่ | หลักฐานในโปรเจกต์ | ความหมาย |
|---|---|---|
| Macro RL 30 steps | `rl/macro_env.py`, `ACTION_DIMS=[5,4,4,4,4]` | ลด horizon จาก 720 แล้ว จึงไม่ใช่ส่วนที่ขาด |
| Macro actions เป็น offsets | ปรับค่ารอบ policy v2.3 | policy ยังผูกกับ fixed rule และออกจากพฤติกรรมเดิมได้ยาก |
| Opponent pool แคบ | v2.2 และ v2.3 | เสี่ยง overfit ต่อคู่แข่งที่เห็นระหว่าง train |
| State เป็น coarse aggregates | macro observation ปัจจุบัน | สูญเสียข้อมูล entity, timing และ market history จนเกิด partial observability ที่หลีกเลี่ยงได้ |
| Reward เป็น proxy | heuristic equity delta + terminal `own_money/70k` + win bonus | อาจ optimize proxy แทนคะแนนการแข่งขันจริง |
| Full-control stack มีของครบหลายส่วน | codec, masks, entity transformer, BC, DAgger, PPO | ควร reuse ไม่ใช่ rebuild |
| ผล v5 ยังต่ำ | mean ราว 45.1k/41.8k และชนะ 6/30 หรือ 4/30 | การมี PPO/BC/DAgger เพียงอย่างเดียวไม่แก้ action semantics, state, reward และ opponent coverage |

ข้อจำกัดของเกมที่สถาปัตยกรรมต้องรองรับพร้อมกันคือ 720 primitive turns, จำนวน hands ที่เปลี่ยนได้, actions แบบมี arguments, illegal actions ที่กลายเป็น silent no-op, ตลาดร่วมที่เปลี่ยนตามคู่แข่ง และข้อมูลคู่แข่งที่เห็นไม่ครบ

## หลักฐานจากงานวิจัย

### 1. Hierarchical และ macro RL

#### Options framework

[Sutton, Precup และ Singh, “Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning,” 1999](https://doi.org/10.1016/S0004-3702(99)00052-1) นิยาม option ด้วย initiation set, intra-option policy และ termination condition และแสดงว่า temporally extended actions สามารถใช้ในกรอบ SMDP และกับวิธีอย่าง Q-learning ได้

การใช้กับ Kaggriculture: ให้ high-level policy เลือก options เช่น `crop_cycle`, `harvest_and_sell`, `prepare_livestock`, `expand_land` หรือ `recover_liquidity` แล้วให้ executor ทำหลาย primitive turns จน option สำเร็จหรือ termination event เกิดขึ้น

ข้อจำกัด: framework ไม่ได้บอกว่า option ใดเหมาะกับโดเมน และสมมติ state แบบ Markov/Semi-Markov ดังนั้น coarse state เดิมอาจทำลายเงื่อนไขสำคัญ

#### Option-Critic

[Bacon, Harb และ Precup, “The Option-Critic Architecture,” 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10916) เรียน intra-option policies, termination functions และ policy over options จาก task reward โดยไม่ต้องเพิ่ม subgoal rewards งานทดลองแสดง end-to-end option learning แต่ยังต้องกำหนดจำนวน options และผู้เขียนใช้ switching penalty ในบางกรณีเพื่อกระตุ้นให้ options ยาวจริง

การใช้กับ Kaggriculture: learned termination เป็นงานระยะหลัง ไม่ใช่จุดเริ่มต้น ควรเริ่มจาก semantic options และ deterministic executor ที่ตรวจสอบได้ก่อน แล้วค่อยทดลองให้ termination head เรียนเมื่อ fixed daily boundary เป็นคอขวด

#### HIRO และความนิ่งของสองระดับ

[Nachum et al., “Data-Efficient Hierarchical Reinforcement Learning,” 2018](https://proceedings.neurips.cc/paper/2018/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html) ชี้ว่าเมื่อ low-level policy เปลี่ยนไป replay ของ high-level policy จะ non-stationary และเสนอ off-policy correction สำหรับ goals

การใช้กับ Kaggriculture: **freeze deterministic executor ระหว่าง train macro policy** ในระยะแรก หากเรียนทั้ง macro และ executor พร้อมกัน ผลลัพธ์ที่แย่อาจมาจาก moving target ไม่ใช่ macro policy เอง

ข้อจำกัด: HIRO ใช้ continuous goals ในงานควบคุมที่ต่างจาก categorical economic goals ของเกม จึงเป็นหลักฐานเรื่อง non-stationarity ไม่ใช่ drop-in algorithm

#### Action branching และ parameterized actions

[Tavakoli, Pardo และ Kormushev, “Action Branching Architectures for Deep Reinforcement Learning,” 2018](https://arxiv.org/abs/1711.08946) ใช้ shared decision module และหนึ่ง branch ต่อ action dimension ทำให้จำนวน network outputs โตเชิงเส้นแทนเชิง combinatorial งานนี้ทดสอบ branch count คงที่และใช้ DQN

[Xiong et al., “Parametrized Deep Q-Networks Learning,” 2018](https://arxiv.org/abs/1810.06394) แยก discrete action type ออกจาก continuous parameters ส่วน [Masson, Ranchod และ Konidaris, “Reinforcement Learning with Parameterized Actions,” 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10226) วางกรอบ parameterized actions รุ่นก่อนหน้า

การใช้กับ Kaggriculture: ใช้แนวคิด **conditional heads** คือเลือก `action_type` ก่อน แล้วเปิดเฉพาะ argument heads ที่เกี่ยวข้อง เช่น item, target และ quantity bucket ไม่ควรนำ P-DQN มาใช้ตรง ๆ เพราะ arguments ส่วนใหญ่ของเกมเป็น categorical/integer และมีข้อจำกัดร่วมกันหลาย hands

ข้อจำกัดสำคัญ: independent branches ไม่รับประกัน joint feasibility ตัวอย่างเช่นหลาย hands อาจจอง seed เดียวกันจนคำสั่งทั้งหมดล้มเหลว จึงต้องมี autoregressive reservation หรือให้ executor จัดสรร resource แบบรวมศูนย์ จำนวน hands ที่เปลี่ยนได้ก็ควรจัดการด้วย shared hand encoder + active mask หรือซ่อนไว้หลัง executor ไม่ใช่สร้าง branch จำนวนตายตัว

### 2. Invalid-action masking

[Huang และ Ontañón, “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms,” 2020/2022](https://arxiv.org/abs/2006.14171) แสดงเชิงทฤษฎีว่า state-dependent masking ให้ policy-gradient update ที่ถูกต้อง และทดลอง PPO ใน µRTS พบว่า masking scale ได้ดีกว่า invalid-action penalty เมื่อสัดส่วน invalid actions เพิ่มขึ้น การถอด mask หลัง train ทำให้ผลแย่ลงเมื่อ action space ใหญ่ขึ้น

ข้อกำหนดสำหรับ Kaggriculture:

- mask logits **ก่อน** softmax ทั้ง train และ evaluation
- mask ต้องครอบคลุม cash, inventory, map boundary, adjacency, ownership, capacity, cooldown และ target type
- หากทุก action ถูก mask ให้เปิด fallback ที่ถูกกฎเสมอ เช่น `PASS`/`HOLD`
- log ทั้ง “raw invalid proposal rate” ก่อน mask และ “executed no-op rate” หลัง executor
- ใช้ joint reservation ledger สำหรับ seed, wheat, cash และ capacity เพราะ per-head masks อย่างเดียวไม่เห็น conflict ระหว่าง hands

ข้อจำกัดของหลักฐาน: paper ทดสอบ PPO ใน resource-collection tasks ของ µRTS ไม่ได้พิสูจน์ผลใน dynamic shared market หรือ simultaneous multi-hand constraints ของ Kaggriculture ดังนั้นต้องมี environment-level legality tests เพิ่มเติม

### 3. Imitation, DAgger และ RL fine-tuning

#### DAgger

[Ross, Gordon และ Bagnell, “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning,” 2011](https://proceedings.mlr.press/v15/ross11a.html) แสดงว่า behavior cloning บน expert-state distribution อาจมี cost เพิ่มระดับ `T²ε` จาก compounding error ขณะที่ DAgger เก็บ states จาก rollout ของ learner, ขอ expert labels บน states เหล่านั้น และ aggregate dataset จนได้ bound ที่โตเชิงเส้นตาม horizon ภายใต้สมมติฐานของงาน

ผลต่อเกม 720 turns: distribution shift เป็นความเสี่ยงหลัก แม้ primitive error จะเล็ก การฝึก imitation ที่ระดับ macro 30 ครั้งจึงสมเหตุผลกว่าระดับ 720 turns

ข้อจำกัดเด็ดขาด: fixed v53 action tape **ไม่ใช่ DAgger expert** เพราะตอบไม่ได้ว่า state ใหม่ที่ learner พาตัวเองเข้าไปควรทำอะไร ต้องสร้าง state-aware teacher จาก planner/rules/recovery logic ก่อน หากไม่มี queryable teacher ให้เรียกขั้นนั้นว่า offline imitation ไม่ใช่ DAgger

[Le et al., “Hierarchical Imitation and Reinforcement Learning,” 2018](https://proceedings.mlr.press/v80/le18a.html) สนับสนุนการ query expert ที่ระดับ subgoal/high level และเสนอ hierarchical DAgger แต่ต้องมี subgoal, termination predicate และ high-level expert ที่นิยามชัด

#### จาก demonstrations ไป RL

[Hester et al., “Deep Q-learning From Demonstrations,” 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11757) pretrain ด้วย TD losses และ large-margin imitation loss แล้วเก็บ demonstration transitions ไว้ถาวรระหว่าง online RL รายงาน initial performance ดีกว่า PDD-DQN ใน 41 จาก 42 Atari games และ pure RL ใช้เฉลี่ยราว 83 ล้าน steps จึงตามทัน

[Rajeswaran et al., “Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations,” 2018](https://www.roboticsproceedings.org/rss14/p49.html) เป็นหลักฐานที่ใกล้ policy-gradient มากกว่า: เริ่มด้วย behavior cloning แล้ว fine-tune ด้วย policy gradient พร้อม demonstration log-likelihood term ที่ลดน้ำหนักตามเวลา

การใช้กับ Kaggriculture: หลัง BC/DAgger ให้ PPO fine-tune พร้อม auxiliary imitation minibatches จาก demonstration buffer และค่อย ๆ ลด coefficient แทนการสลับจาก imitation เป็น RL แบบทันที นี่เป็นการประยุกต์จาก DQfD/DAPG ไม่ใช่การรับประกันโดยตรงสำหรับ recurrent PPO จึงต้องทำ ablation

### 4. Robustness ต่อคู่แข่งและ self-play

[Lanctot et al., “A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning,” 2017](https://arxiv.org/abs/1711.00832) พบว่า independent RL สามารถ overfit ต่อ policies ที่พบตอน train และเสนอ Policy-Space Response Oracles (PSRO): ฝึก approximate best response ต่อ mixture ของ policies แล้วใช้ empirical payoff game คำนวณ meta-strategy งานทดลองรวม partially observable coordination games และ imperfect-information poker

[Vinyals et al., “Grandmaster Level in StarCraft II Using Multi-agent Reinforcement Learning,” 2019](https://doi.org/10.1038/s41586-019-1724-z) ใช้ข้อมูลทั้ง human และ agent games ใน diverse league ที่สร้าง strategies และ counter-strategies อย่างต่อเนื่อง เป็นหลักฐานว่าการผสม imitation กับ population-based self-play เหมาะกับเกมยาวที่มี structured interactions และคู่แข่งเปลี่ยนตลอด

การใช้กับ Kaggriculture แบบประหยัดกว่าระบบเต็ม:

- เก็บ frozen snapshots ของ current policy และ historical best
- รวม rule bots v2.2, v2.3, v53 และ recovery/adversarial market bots
- ฝึก specialized exploiters ที่เน้นกดราคา, แย่งซื้อ หรือเร่ง expansion
- สุ่มคู่แข่งจาก mixture ที่มี full support ไม่ใช่เล่นกับ current self อย่างเดียว
- แยก held-out opponents ออกจาก training league และวัด cross-play matrix

ข้อจำกัด: full PSRO ต้องฝึก best responses หลายตัวและประเมิน payoff matrix ซึ่งแพงมาก approximate best responses ก็ไม่รับประกัน Nash convergence เกมนี้อาจไม่ใช่ two-player zero-sum อย่างเคร่งครัด จึงไม่ควรอ้าง exploitability หากยังไม่พิสูจน์ payoff structure ให้ใช้ held-out win rate, relative score และ worst-case quantile แทน

### 5. Partial observability และ memory

[Hausknecht และ Stone, “Deep Recurrent Q-Learning for Partially Observable MDPs,” 2015](https://arxiv.org/abs/1507.06527) แสดงว่า LSTM สามารถรวมข้อมูลข้ามเวลาใน flickering Atari ได้ แต่ผู้เขียนพบว่า recurrence ไม่มี systematic advantage เหนือ stacked observations ในเกมส่วนใหญ่

ลำดับที่ถูกต้องสำหรับ Kaggricultureจึงเป็น:

1. เลิกทิ้งข้อมูล public state ที่จำเป็นก่อน: entities, hand status, inventory, structures, crop ages, prices, recent market transactions, day/turn และ opponent-visible actions
2. reuse entity transformer/full-control encoder ที่มีอยู่และ pool active hands แบบ permutation-aware
3. เพิ่ม GRU/LSTM ที่ระดับ macro เฉพาะข้อมูลที่ต้องอนุมานจาก history เช่น opponent intent และ market trend
4. ทำ recurrence ablation เทียบกับ frame stacking เพราะ memory ไม่แก้ state representation ที่ออกแบบผิด

### 6. Reward ที่ไม่เปลี่ยนเป้าหมาย

[Ng, Harada และ Russell, “Policy Invariance under Reward Transformations,” 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) แสดงว่า potential-based shaping

`F(s,a,s') = γΦ(s') - Φ(s)`

เป็นเงื่อนไขที่รักษา optimal policies ภายใต้สมมติฐานของ finite MDP และเป็นรูปแบบเดียวที่รับประกันได้โดยไม่เพิ่มสมมติฐานอื่น

คำแนะนำ:

- terminal reward ต้องตรงกับ Kaggle scoring objective หรือ relative competition outcome จริง
- หากต้องการ dense reward ให้แปลง heuristic equity เป็น potential `Φ` แล้วใช้เฉพาะ potential difference แทนการบวก equity delta ที่ออกแบบตามใจ
- กำหนด terminal/absorbing potential อย่างสอดคล้องและตรวจ reward decomposition ทุก episode

ข้อจำกัด: theorem เดิมเป็น MDP หากใช้ observation ที่สูญเสียข้อมูลหรือมี opponent non-stationarity การรับประกันไม่ย้ายมาตรง ๆ นี่เป็นเหตุผลเพิ่มอีกข้อให้แก้ state และ opponent training ก่อนเชื่อ reward shaping

## สถาปัตยกรรมที่เสนอ

```text
public entity/economy state + recent history
                  │
       entity encoder + optional GRU
                  │
      high-level semantic option policy
                  │
   masked conditional argument/target heads
                  │
   deterministic planner + reservation ledger
                  │
  legal primitive actions for all active hands
          (replan every primitive turn)
```

### High-level policy

ตัดสินใจที่ day boundary และอนุญาต event-triggered termination/replan เมื่อ option สำเร็จ, เงิน/วัตถุดิบขาด, market threshold เปลี่ยน หรือ action กลายเป็น illegal

ให้ policy ออก **targets ไม่ใช่ offsets** เช่น:

- production allocation และ crop targets
- target animal/structure counts
- target worker count และ role priorities
- cash reserve, wheat reserve และ sell urgency
- land-expansion budget/intent
- recovery/liquidity mode

สามารถเริ่มด้วยขนาด `ACTION_DIMS` ใกล้ของเดิมเพื่อควบคุมความเสี่ยง แต่แต่ละ bucket ต้องมีความหมายเชิง state ที่สมบูรณ์ ไม่ใช่ `old_rule_action + delta`

### Deterministic executor

Executor รับ target แล้วจัดเส้นทาง, pickup/drop, resource reservation, assignment ของ hands, legality และ market orders ใหม่จาก state ทุก primitive turn จึงไม่ใช่ fixed tape แม้จะ deterministic

ผลลัพธ์ทุก option ควรคืน `success`, `progress`, `blocked_reason`, `fallback_count` และ `turns_used` เพื่อให้ policy เห็นว่ากลยุทธ์ใดทำจริงไม่ได้

## แผนฝึกที่แนะนำ

1. **Semantic-action baseline** — เปลี่ยน macro offsets เป็น state-aware targets แต่คง PPO, executor และ opponent set เพื่อแยกผลของ action design
2. **Macro BC** — แปลง replay ที่ดีเป็น macro-labelled trajectories; ฝึก high-level heads เท่านั้น
3. **Macro DAgger** — rollout learner บน seeds/opponents หลากหลาย แล้วให้ state-aware teacher label states ที่ learner พบ หาก teacher ตอบ state ใหม่ไม่ได้ให้หยุดที่ BC
4. **Masked recurrent PPO** — fine-tune ด้วย terminal objective + potential shaping; คง demonstration auxiliary loss แบบลดน้ำหนัก
5. **PSRO-lite league** — เพิ่ม historical policies และ exploiters ทีละชุด; freeze evaluation pool
6. **Optional learned termination** — ทดลอง Option-Critic-style termination หลัง semantic/executor baseline ผ่านแล้วเท่านั้น

## การประเมินและ stop conditions

อย่าใช้ public Kaggle score เป็น training loop หลัก ให้สร้างชุดทดสอบในเครื่องที่ไม่ปนกับ train:

- held-out seeds และ held-out opponent policies
- cross-play matrix ระหว่าง v53, current v5, BC, DAgger, DAgger+PPO และ league policy
- รายงาน mean, median, lower decile, worst-case, win rate และ relative score
- รายงาน raw invalid proposal rate, post-mask no-op rate, option success rate และ executor fallback rate
- bootstrap confidence intervals หรือหลาย independent training seeds

Ablation ที่ต้องทำแยกกัน:

1. offsets เทียบ semantic targets
2. coarse state เทียบ enriched entity/economy state
3. heuristic reward เทียบ terminal + potential shaping
4. BC เทียบ BC+DAgger
5. feed-forward เทียบ recurrent
6. fixed opponents เทียบ league

หยุดหรือย้อนการเปลี่ยนแปลงหากค่าเฉลี่ยดีขึ้นแต่ lower-decile/worst-case แย่ลง, no-op เพิ่มขึ้น, หรือผลดีมีเฉพาะคู่แข่งใน training pool เพราะนั่นไม่ใช่ความยั่งยืน

## ลำดับความสำคัญ

1. เปลี่ยน action semantics, state และ reward ใน macro environment เดิม
2. สร้าง queryable state-aware teacher และ deterministic executor ที่ freeze ได้
3. reuse masks/entity encoder/BC/DAgger/PPO ที่มีอยู่
4. เพิ่ม opponent league และ held-out cross-play
5. พิจารณา learned low-level/termination เฉพาะเมื่อมีหลักฐานว่า executor เป็นคอขวด

ข้อสรุปสุดท้าย: **RL ควรเรียน “จะทำอะไรและเมื่อไร” ส่วน executor รับผิดชอบ “ทำอย่างไรให้ถูกกฎ”** นี่ลด horizon, ป้องกัน silent no-ops, รองรับจำนวน hands ที่เปลี่ยนได้ และยังตอบสนองตลาดกับคู่แข่งจาก state ปัจจุบันโดยไม่ย้อนกลับไปเป็น fixed action tape
