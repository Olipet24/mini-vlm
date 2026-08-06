# New-Data Collection Guide (final report, "Evaluate on New Data")

This is the checklist + question bank for the 10-point "evaluate on new data" rubric item.
**Every answer below is an exact string from the model's fixed 1000-answer vocabulary**
(`outputs/processed_full/answer_vocab.json`) — verified programmatically, not guessed. You do
not need to change any answer text; just take the photo described and it should match.

## Rules (read once, then just follow the table)

1. **Take all 50 photos first**, save them as `photo01.jpg` ... `photo50.jpg` in
   `data_new/images/` (any resolution/orientation is fine — resizing and EXIF rotation are
   handled automatically).
2. **Write/confirm answers before running the model on them.** Never look at a prediction and
   then edit a ground-truth answer, and never change a hyperparameter after seeing these
   results — this discipline is exactly what the rubric's top band asks for, and I'll state it
   explicitly in the report.
3. Build `data_new/questions.csv` from the table below (columns: `image_filename,question,
   answer,answer_type` — copy the table rows directly, they're already in that order).
4. Run the validator **early**, after your first 10-15 photos, not after all 50:
   ```
   python -m mini_vlm.eval_new_data --validate-only \
       --images-dir data_new/images --questions data_new/questions.csv \
       --processed-dir outputs/processed_full
   ```
   Fix anything it flags before continuing — it's much faster to fix 2 rows than 100.
5. Once all 50 photos + `questions.csv` are ready, tell me and I'll run the full evaluation.

Mix: 40 yes/no, 40 other, 20 number (100 questions total) — matches the training distribution
so the per-type comparison in the report is apples-to-apples. Photos 1-40 each answer one
yes/no + one "other" question; photos 41-50 are dedicated counting shots (2 number questions
each) — for those, arrange the exact quantities listed, they're the ground truth.

## Photos to take (1-40: single-object/scene shots)

| # | What to photograph |
|---|---|
| 1 | A red apple on a plain surface |
| 2 | A green apple on a plain surface |
| 3 | A banana |
| 4 | A cup with coffee in it |
| 5 | A laptop on a desk |
| 6 | A pair of sunglasses |
| 7 | An umbrella |
| 8 | A backpack |
| 9 | A bicycle |
| 10 | A wristwatch |
| 11 | A blue shirt or piece of blue clothing |
| 12 | A black shirt or piece of black clothing |
| 13 | A pair of shoes |
| 14 | A book |
| 15 | A pillow |
| 16 | A clock (wall clock or any visible clock) |
| 17 | A mirror |
| 18 | A candle |
| 19 | A vase (empty or with flowers) |
| 20 | A vase with flowers in it |
| 21 | A potted plant |
| 22 | A TV/AC remote control |
| 23 | A cell phone |
| 24 | A computer mouse |
| 25 | A computer keyboard |
| 26 | Your kitchen, with the microwave visible |
| 27 | Your bathroom, with the sink visible |
| 28 | Your bedroom, with the bed visible |
| 29 | Your living room, with a couch visible |
| 30 | A slice of pizza |
| 31 | A sandwich |
| 32 | A bowl of salad |
| 33 | A slice of cake |
| 34 | A cup with tea in it |
| 35 | A glass of milk |
| 36 | A car (yours or any car outside) |
| 37 | The sky when it's cloudy/overcast |
| 38 | The sky when it's sunny/clear |
| 39 | A patch of grass or lawn |
| 40 | A tree |

## Photos to take (41-50: counting shots — arrange these exact quantities)

| # | What to arrange and photograph |
|---|---|
| 41 | Exactly 3 apples and 2 bananas together in one frame |
| 42 | Exactly 4 books (stacked or in a row) and 1 pen next to them |
| 43 | Exactly 5 forks and 2 spoons on a table |
| 44 | Exactly 6 chairs (or 6 of any identical small object) and an empty plate with 0 donuts on it |
| 45 | Exactly 7 books on a shelf and 1 clock nearby |
| 46 | Exactly 8 bottles lined up and 2 cups next to them |
| 47 | Exactly 10 grapes and 3 apples together in one frame |
| 48 | A room shot showing exactly 2 windows and 1 door |
| 49 | Exactly 4 pillows and 1 blanket on a bed or couch |
| 50 | Exactly 5 oranges in a bowl and 6 strawberries next to the bowl |

## Question bank (100 rows — copy directly into questions.csv)

`image_filename,question,answer,answer_type`

```
photo01.jpg,Is this a red apple?,yes,yes/no
photo01.jpg,What color is the apple?,red,other
photo02.jpg,Is the apple red?,no,yes/no
photo02.jpg,What color is the apple?,green,other
photo03.jpg,Is this a banana?,yes,yes/no
photo03.jpg,What color is the banana?,yellow,other
photo04.jpg,Is there coffee in the cup?,yes,yes/no
photo04.jpg,What is in the cup?,coffee,other
photo05.jpg,Is this a laptop?,yes,yes/no
photo05.jpg,What is on the desk?,laptop,other
photo06.jpg,Are these sunglasses?,yes,yes/no
photo06.jpg,What is this object?,sunglasses,other
photo07.jpg,Is this an umbrella?,yes,yes/no
photo07.jpg,What is this?,umbrella,other
photo08.jpg,Is this a backpack?,yes,yes/no
photo08.jpg,What is this object?,backpack,other
photo09.jpg,Is this a bicycle?,yes,yes/no
photo09.jpg,What is this?,bicycle,other
photo10.jpg,Is this a watch?,yes,yes/no
photo10.jpg,What is this object?,watch,other
photo11.jpg,Is the shirt blue?,yes,yes/no
photo11.jpg,What color is the shirt?,blue,other
photo12.jpg,Is the shirt white?,no,yes/no
photo12.jpg,What color is the shirt?,black,other
photo13.jpg,Are these shoes?,yes,yes/no
photo13.jpg,What are these?,shoes,other
photo14.jpg,Is this a book?,yes,yes/no
photo14.jpg,What is this object?,book,other
photo15.jpg,Is this a pillow?,yes,yes/no
photo15.jpg,What is this?,pillow,other
photo16.jpg,Is this a clock?,yes,yes/no
photo16.jpg,What is on the wall?,clock,other
photo17.jpg,Is this a mirror?,yes,yes/no
photo17.jpg,What is this object?,mirror,other
photo18.jpg,Is this a candle?,yes,yes/no
photo18.jpg,What is this?,candle,other
photo19.jpg,Is this a vase?,yes,yes/no
photo19.jpg,What is this object?,vase,other
photo20.jpg,Are there flowers in the photo?,yes,yes/no
photo20.jpg,What is in the vase?,flowers,other
photo21.jpg,Is this a plant?,yes,yes/no
photo21.jpg,What is this?,plant,other
photo22.jpg,Is this a remote?,yes,yes/no
photo22.jpg,What is this object?,remote,other
photo23.jpg,Is this a phone?,yes,yes/no
photo23.jpg,What is this object?,cell phone,other
photo24.jpg,Is this a mouse?,yes,yes/no
photo24.jpg,What is this object?,mouse,other
photo25.jpg,Is this a keyboard?,yes,yes/no
photo25.jpg,What is this?,keyboard,other
photo26.jpg,Is this a microwave?,yes,yes/no
photo26.jpg,What room is this?,kitchen,other
photo27.jpg,Is this a bathroom?,yes,yes/no
photo27.jpg,What room is this?,bathroom,other
photo28.jpg,Is there a bed in this room?,yes,yes/no
photo28.jpg,What room is this?,bedroom,other
photo29.jpg,Is there a couch in this room?,yes,yes/no
photo29.jpg,What room is this?,living room,other
photo30.jpg,Is this pizza?,yes,yes/no
photo30.jpg,What food is this?,pizza,other
photo31.jpg,Is this a sandwich?,yes,yes/no
photo31.jpg,What food is this?,sandwich,other
photo32.jpg,Is this a salad?,yes,yes/no
photo32.jpg,What food is this?,salad,other
photo33.jpg,Is this cake?,yes,yes/no
photo33.jpg,What food is this?,cake,other
photo34.jpg,Is this tea?,yes,yes/no
photo34.jpg,What is in the cup?,tea,other
photo35.jpg,Is this milk?,yes,yes/no
photo35.jpg,What is in the glass?,milk,other
photo36.jpg,Is this a car?,yes,yes/no
photo36.jpg,What is this vehicle?,car,other
photo37.jpg,Is it sunny?,no,yes/no
photo37.jpg,What is the weather like?,cloudy,other
photo38.jpg,Is it sunny?,yes,yes/no
photo38.jpg,What is the weather like?,sunny,other
photo39.jpg,Is there grass in the photo?,yes,yes/no
photo39.jpg,What is on the ground?,grass,other
photo40.jpg,Is this a tree?,yes,yes/no
photo40.jpg,What is this?,tree,other
photo41.jpg,How many apples are there?,3,number
photo41.jpg,How many bananas are there?,2,number
photo42.jpg,How many books are there?,4,number
photo42.jpg,How many pens are there?,1,number
photo43.jpg,How many forks are there?,5,number
photo43.jpg,How many spoons are there?,2,number
photo44.jpg,How many chairs are there?,6,number
photo44.jpg,How many donuts are on the plate?,0,number
photo45.jpg,How many books are on the shelf?,7,number
photo45.jpg,How many clocks are there?,1,number
photo46.jpg,How many bottles are there?,8,number
photo46.jpg,How many cups are there?,2,number
photo47.jpg,How many grapes are there?,10,number
photo47.jpg,How many apples are there?,3,number
photo48.jpg,How many windows are there?,2,number
photo48.jpg,How many doors are there?,1,number
photo49.jpg,How many pillows are there?,4,number
photo49.jpg,How many blankets are there?,1,number
photo50.jpg,How many oranges are there?,5,number
photo50.jpg,How many strawberries are there?,6,number
```

## Quick tally

- yes/no: 40 (photos 1-40, one each)
- other: 40 (photos 1-40, one each)
- number: 20 (photos 41-50, two each)
- **Total: 100 questions across 50 photos**
