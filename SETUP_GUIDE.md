# AI Code Reviewer - Setup Guide

## ⚠️ Prerequisites - Dataset Download Required

**Before running setup**, download and prepare datasets:

1. Download from: https://zenodo.org/records/6900648
2. Create `Datasets/` folder in repository root
3. Place these two datasets inside `Datasets/`:
   - `Code_Refinement/` (ref-train.jsonl, ref-valid.jsonl, ref-test.jsonl)
   - `Comment_Generation/` (msg-train.jsonl, msg-valid.jsonl, msg-test.jsonl)

Expected structure:
```
Datasets/
├── Code_Refinement/
│   ├── ref-train.jsonl
│   ├── ref-valid.jsonl
│   └── ref-test.jsonl
├── Comment_Generation/
│   ├── msg-train.jsonl
│   ├── msg-valid.jsonl
│   └── msg-test.jsonl
└── Unified_Dataset/  (created by setup)
```

---

## Quick Setup

### macOS

```bash
cd /path/to/AI-Code-Reviewer
chmod +x setup_indexing.sh
./setup_indexing.sh
```

Or without chmod:
```bash
bash setup_indexing.sh
```


---

### Windows (Git Bash - Recommended)

```bash
bash setup_indexing.sh
```

---

## Common Issues

**Python < 3.10:** Upgrade to Python 3.10+  
**PowerShell blocks scripts:** `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`  
**Datasets not found:** Follow Prerequisites section  
**Permission denied on macOS:** `chmod +x setup_indexing.sh`

---

## Daily Use

**Activate venv:**
- macOS/Linux/Git Bash: `source .venv/bin/activate`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows CMD: `.\.venv\Scripts\activate`

**Deactivate:** `deactivate`
