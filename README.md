## Question 1

### Run Q1_part1.py to see stats for emission parameters

**Without smoothing:**

```bash
python Q1_part1.py
```

**With smoothing:**

```bash
python .\Q1_part1.py --smoothing --k 3
```

### Run Q1_part2.py to output ‘dev.p1.out’ for evaluation

```bash
python Q1_part2.py
```

### Run evaluation script:
```bash
python evalResult.py EN/dev.out EN/dev.p1.out
```