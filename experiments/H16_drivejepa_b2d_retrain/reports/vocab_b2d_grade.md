# Vocabulary size on Bench2Drive grading (05 Nov 2026)

Replaces the ADE-based 1,024 pick (Navvrat: "ade is not a metric, b2d scores are"). Every feasible path of each vocabulary was graded offline with `grade.py`: route share x 0.6^collision x 0.7^red x (1 - share outside lanes or wrong way). Teacher = grade >= 0.95. 298 held-out frames (39 of 47 clips), 299 training frames (100 clips). Full tables: `analyse.txt`, `analyse_turns.txt`. Intervals are 95 % clip bootstraps.

| Size | Held-out best grade | Held-out >= 1 teacher | Held-out >= 4 | Held-out turns >= 1 | Held-out turns >= 4 | Train >= 1 | Train >= 4 | Train turns >= 4 | Median teachers, held-out | Entries from one clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | .966 | .862 | .711 | .451 | .255 | .930 | .763 | .340 | 13 | 0.7 % |
| 1,024 | .984 | .930 | .772 | .686 | .294 | .970 | .849 | .574 | 22 | 2.1 % |
| 2,048 | .989 | .956 | .842 | .784 | .392 | .990 | .916 | .745 | 42 | 7.3 % |
| 4,096 | .992 | .966 | .906 | .843 | .588 | .990 | .946 | .809 | 85 | 22.3 % |
| 8,192 | .995 | .980 | .943 | .882 | .745 | .997 | .970 | .915 | 168 | 50.2 % |
| NAVSIM 8,192 | .942 | .775 | .735 | .412 | .333 | .839 | .759 | | 270 | |

Expert's own grade: .996 held-out, 1.000 train.

## What the numbers show

- All of the difference between sizes is on turns (51 held-out frames, 21 clips). Straight and stopped frames have a teacher on .977+ of frames from 1,024 up.
- Held-out steps, >= 1 teacher: 512 to 1,024 +.067 [+.040, +.099]; 1,024 to 2,048 +.027 [+.013, +.044]; 2,048 to 4,096 +.010 [-.007, +.030]; 4,096 to 8,192 +.013 [.000, +.031].
- Held-out steps, >= 4 teachers: +.060, +.070, +.064 [+.030, +.100], +.037 [+.017, +.060]. On held-out turns 2,048 to 4,096 is +.196 [+.074, +.308].
- Training uses the training frames, so the train columns are the actual teacher supply. Train turns >= 4, 2,048 to 4,096: +.064 [.000, +.143].
- Teacher share of feasible paths is flat, about 10 % at every size. Mean teacher spread is flat, about 3.0 m. Size adds count, not spread.
- Six held-out turns have no teacher even at 8,192: tight left turns at speed, best grade .59 to .94.
- Vocabularies are not nested: 2 held-out frames lose their teachers going to a larger size.

## Limits

This is a stand-in for the unbuilt grader, not a driving score. No stop-sign rule; 0.6 for every collision; log replay; 4 s local grade, not a route. Frames were sampled evenly per clip. Some of the teacher count at 4,096 and 8,192 is near-copies (entries within 1 m: 15 / 28 / 52 at 2,048 / 4,096 / 8,192), so ">= 4" does not mean 4 distinct paths. The rows hold counts, not indices, so unused entries cannot be counted without a rerun.
