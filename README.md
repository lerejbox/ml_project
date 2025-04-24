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

### Run Q1_part2.py to output ‘dev.p2.out’ for evaluation

```bash
python Q1_part2.py
```

### Run evaluation script:
```bash
python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out
```

### Question 2

### Run Q2_part1.py to see transition parameters

```bash
python Q2_part1.py
```

### Run Q2_part2.py to output 'dev.p2.out' for evaluation

```bash
python Q2_part2.py
```

### Run evaluation script:

```bash
python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out
```

### Run Q3.py to output 'dev.p3.out' for evaluation

```bash
python Q3.py
```