import sqlparse
import sqlite3

# -----------------------
# 1. Load GT & Prediction
# -----------------------
gt_sql_path = "data/dev.sql"
pred_sql_path = "results/t5_ft_experiment_dev.sql"

gt_sql = [l.strip() for l in open(gt_sql_path)]
pred_sql = [l.strip() for l in open(pred_sql_path)]

assert len(gt_sql) == len(pred_sql), "Dev SQL count mismatch."

total = len(gt_sql)
print(f"Total dev examples = {total}\n")

# -----------------------
# 2. Helper functions
# -----------------------

def has_join(sql):
    return " join " in sql.lower()

def get_columns(sql):
    tokens = sql.lower().replace(",", " ").split()
    cols = []
    for t in tokens:
        if "_" in t:  # crude but works well for this dataset
            cols.append(t)
    return set(cols)

def sql_syntax_ok(sql):
    try:
        sqlite3.connect(":memory:").execute(f"EXPLAIN {sql}")
        return True
    except:
        return False

def compare_where(gt, pred):
    if "where" not in gt.lower() or "where" not in pred.lower():
        return False
    # simple check on comparison operators
    wrong_ops = ["<", ">", "<=", ">=", "="]
    return any(op in gt and op not in pred for op in wrong_ops)

# -----------------------
# 3. Error Buckets
# -----------------------
errors = {
    "Missing JOIN": [],
    "Wrong Column": [],
    "Incorrect WHERE": [],
    "Syntax Error": [],
    "Other": []
}

# -----------------------
# 4. Classification Loop
# -----------------------
for i, (gt, pred) in enumerate(zip(gt_sql, pred_sql)):
    if gt.lower() == pred.lower():
        continue  # correct

    # 1. Syntax error check
    if not sql_syntax_ok(pred):
        errors["Syntax Error"].append((i, gt, pred))
        continue

    # 2. Missing join
    if has_join(gt) and not has_join(pred):
        errors["Missing JOIN"].append((i, gt, pred))
        continue

    # 3. Wrong column
    gt_cols = get_columns(gt)
    pred_cols = get_columns(pred)
    if not gt_cols.issubset(pred_cols):
        errors["Wrong Column"].append((i, gt, pred))
        continue

    # 4. Wrong where condition
    if compare_where(gt.lower(), pred.lower()):
        errors["Incorrect WHERE"].append((i, gt, pred))
        continue

    # 5. everything else
    errors["Other"].append((i, gt, pred))

# -----------------------
# 5. Print results
# -----------------------
for k, v in errors.items():
    print(f"{k}: {len(v)} errors out of {total}")

print("\n===== Sample Examples =====")
for k, v in errors.items():
    print(f"\n### {k} (showing up to 2 examples)")
    for item in v[:2]:
        idx, g, p = item
        print(f"\n[ID {idx}]")
        print("GT:   ", g)
        print("Pred: ", p)
