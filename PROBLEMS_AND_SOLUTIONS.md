# 🐛 Problems & Solutions Log

> This file tracks all bugs, errors, and issues encountered during development.
> Every problem is documented with its solution for future reference.

---

## Problem #1 — `uvicorn` command not recognized

| Field | Details |
|-------|---------|
| **Date** | 2026-04-05 |
| **Phase** | Running API Server |
| **Severity** | 🔴 Blocker |

**Error:**
```
uvicorn : The term 'uvicorn' is not recognized as the name of a cmdlet, function, script file, or operable program.
    + CategoryInfo : ObjectNotFound: (uvicorn:String) [], CommandNotFoundException
```

**Cause:**
pip installed packages to user site-packages (`C:\Users\joy\AppData\Local\Packages\...\Python312\Scripts`), which is **not on the system PATH**. So CLI tools like `uvicorn` can't be found directly.

**Solution:**
Use `python -m` prefix to run the module directly instead of relying on PATH:
```powershell
python -m uvicorn app.main:app --reload
```

**Status:** ✅ Fixed

---

## Problem #2 — `streamlit` command not recognized

| Field | Details |
|-------|---------|
| **Date** | 2026-04-05 |
| **Phase** | Running Streamlit Frontend |
| **Severity** | 🔴 Blocker |

**Error:**
```
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet, function, script file, or operable program.
    + CategoryInfo : ObjectNotFound: (streamlit:String) [], CommandNotFoundException
```

**Cause:**
Same as Problem #1 — Python Scripts directory is not on PATH.

**Solution:**
Use `python -m` to run streamlit:
```powershell
python -m streamlit run frontend/app.py
```

**Status:** ✅ Fixed

---

## Problem #3 — Missing Data Files

| Field | Details |
|-------|---------|
| **Date** | 2026-04-10 |
| **Phase** | Model Training |
| **Severity** | 🔴 Blocker |

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../data/emails.csv'
```

**Cause:**
Training script expects CSV data files that weren't generated.

**Solution:**
Generate synthetic training data via `notebooks/train.py` if files don't exist.

**Status:** ✅ Fixed

