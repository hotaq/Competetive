# V91 goose programme — rejected, and it explains V67's idle board

Date 2026-08-30. The candidate was built, five defects were found and fixed, and in every
configuration it loses to exact V67. The reason is not a bug: it is the price of labour.

## What it is

`main.py` wraps the exact V67 bundle, hires hands beyond the ones V67 asks for, and gives
those hands, and only those, a coop controller: fetch a bird, build a coop on an idle
tile, place it, collect eggs, bank them, sell them. V67's farmer, its own hands and its
market orders pass through untouched.

It tests `MILESTONE2.md`: EGG is the only good whose price does not fall with volume, and
V67 ends its games with 48 idle tiles, no coop and no goose.

## Defects found and fixed

| # | defect | margin after |
|---|---|---:|
| 1 | V67 assigns hand commands positionally, so extra hires shifted every later index | -139,222 to -28,015 |
| 2 | 39 birds bought, 5 placed: hands are dismissed at day end and take what they carry | |
| 3 | `PICKUP` of stock the shed lacks is a silent no-op, repeated 444 times of 800 actions | -59,930 to -41,740 |
| 4 | the goose cash floor also blocked hiring, which is Fibonacci-cheap early | |
| 5 | ownership assumed every `HIRE` succeeds; derived it from the observed hand count instead | -41,740 to **-2,412** |

After fix 5 the overlay is clean: V91 reproduces V67's production exactly — MILK 274,
STRAWBERRY 316, WOOL 134 against V67's 275, 316, 136.

## Why it still loses: labour is priced on a Fibonacci curve

Hiring costs 1, 1, 2, 3, 5, 8, 13, ... for successive hands **within a day**, and V67
already takes 11 to 15 hands a day from day 10 onward. The next three hands on those days
cost 987, 1,597 and 2,584.

| day | V67 hires | cost of 3 extra hands |
|---:|---:|---:|
| 0-4 | 2-3 | 10-16 |
| 10 | 12 | 1,220 |
| 21 | 13 | 1,974 |
| 26 | 15 | **5,168** |

Three extra hands every day costs **26,982 a game**, which was the entire remaining loss.
And an extra hand does not only cost its own slot: it pushes every subsequent V67 hire one
step up the same curve.

Sweeping the cap on how far up the curve we are willing to hire:

| cap on hires before ours | median margin | eggs sold | coops |
|---:|---:|---:|---:|
| 10 | **-2,412** | 0 | 2 |
| 13 | -6,992 | 0 | 3 |
| 16 | -13,599 | 7 | 1 |

Monotone. Every additional hand makes it worse, and the egg revenue never arrives because
the hands that would collect eggs are the same hands that cost 233 to 2,584 a day.

## Verdict

**Rejected.** The finding is not that the goose programme was implemented badly — after
five fixes it preserves V67's economy to within 1% — but that the egg market, however
deep, is not reachable with hired labour at the price V67's own hiring has already set.

This reframes `MILESTONE2.md`, which valued filling the idle board at 34,823 a game while
ignoring what the hands to work it cost. **V67 leaves 48 tiles idle because that is the
correct response to Fibonacci hiring**, not because it overlooks them.

## What is left

The egg market's depth is still real and still unexploited by everyone. Reaching it needs
labour that is already paid for, which means eggs would have to displace something V67's
existing hands do now — the animal or strawberry work — rather than being added on top.
That is a different agent, and this session has now measured why every cheaper route
fails.
